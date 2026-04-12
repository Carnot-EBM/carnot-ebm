#!/usr/bin/env python3
"""Experiment 210: Constraint extraction research scan for instruction-tuned models.

This workflow materializes a curated literature scan into:

- ``results/experiment_210_results.json``
- ``research-references.md``
- ``research-studying.md``

The scan is intentionally deterministic at runtime: the papers and proposals
were curated from the current literature review and are embedded below so the
repo can refresh the artifact and markdown sections without depending on live
network access.

Spec: REQ-REPORT-005, REQ-REPORT-006, REQ-REPORT-007, REQ-REPORT-008,
SCENARIO-REPORT-004, SCENARIO-REPORT-005
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


REPO_ROOT = get_repo_root()
RESULTS_PATH = REPO_ROOT / "results" / "experiment_210_results.json"
REFERENCES_PATH = REPO_ROOT / "research-references.md"
STUDYING_PATH = REPO_ROOT / "research-studying.md"

REFERENCES_START_MARKER = "<!-- EXP210_REFERENCES_START -->"
REFERENCES_END_MARKER = "<!-- EXP210_REFERENCES_END -->"
STUDYING_START_MARKER = "<!-- EXP210_STUDYING_START -->"
STUDYING_END_MARKER = "<!-- EXP210_STUDYING_END -->"

QUERY_THEMES = [
    "constraint extraction from chain-of-thought responses",
    "formal verification of LLM reasoning steps",
    "SMT/SAT-based LLM output verification",
    "neuro-symbolic verification frameworks",
    "instruction-following constraint decomposition benchmarks",
    "chain-of-thought monitorability and obfuscation risk",
]

PAPERS: list[dict[str, Any]] = [
    {
        "rank": 1,
        "title": "Neuro-Symbolic Verification on Instruction Following of LLMs",
        "year": 2026,
        "source": "arXiv 2601.17789",
        "category": "instruction-following verification",
        "url": "https://arxiv.org/abs/2601.17789",
        "why_it_matters": "Most direct fit to Carnot's current blocker: it treats instruction following as a constraint-satisfaction problem and combines logical plus semantic checks in one verifier.",
        "recommended_carnot_use": "Use as the primary template for a prompt-to-constraint intermediate representation plus solver-backed verification path.",
    },
    {
        "rank": 2,
        "title": "ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming",
        "year": 2025,
        "source": "EMNLP 2025",
        "category": "natural-language-to-CP modeling",
        "url": "https://aclanthology.org/2025.emnlp-main.809/",
        "why_it_matters": "Shows that instruction-tuned models can be specialized for constraint programming, paired with retrieval and guided self-correction, and evaluated on an industrial benchmark.",
        "recommended_carnot_use": "Borrow the CP modeling pattern for scheduling and resource constraints, especially as a second solver route beside Z3.",
    },
    {
        "rank": 3,
        "title": "LLM Self-Correction with DeCRIM: Decompose, Critique, and Refine for Enhanced Following of Instructions with Multiple Constraints",
        "year": 2024,
        "source": "Findings of EMNLP 2024",
        "category": "multi-constraint decomposition",
        "url": "https://aclanthology.org/2024.findings-emnlp.458/",
        "why_it_matters": "Directly decomposes instructions into atomic constraints, then critiques failures at the constraint level using RealInstruct and IFEval.",
        "recommended_carnot_use": "Build Carnot's first prompt-side atomic constraint extractor and use DeCRIM-style critique labels as supervision.",
    },
    {
        "rank": 4,
        "title": "CARE-STaR: Constraint-aware Self-taught Reasoner",
        "year": 2025,
        "source": "Findings of ACL 2025",
        "category": "constraint difficulty decomposition",
        "url": "https://aclanthology.org/2025.findings-acl.1116/",
        "why_it_matters": "Separates easy versus ambiguous constraints and learns different reasoning traces for different constraint levels.",
        "recommended_carnot_use": "Route high-confidence literal constraints to symbolic solvers and keep ambiguous constraints on a softer verification path.",
    },
    {
        "rank": 5,
        "title": "VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks",
        "year": 2025,
        "source": "arXiv 2511.04662",
        "category": "CoT formal verification",
        "url": "https://arxiv.org/abs/2511.04662",
        "why_it_matters": "Formalizes each chain-of-thought step into first-order logic and checks whether each step is grounded in source context, commonsense, or prior steps.",
        "recommended_carnot_use": "Prototype a typed step graph for arithmetic and logic traces instead of relying on raw free-form chain-of-thought text.",
    },
    {
        "rank": 6,
        "title": "PCRLLM: Proof-Carrying Reasoning with Large Language Models under Stepwise Logical Constraints",
        "year": 2025,
        "source": "arXiv 2511.08392",
        "category": "stepwise proof-carrying reasoning",
        "url": "https://arxiv.org/abs/2511.08392",
        "why_it_matters": "Constrains each reasoning step to explicit premises, rules, and conclusions so chain-level validation becomes possible even for black-box models.",
        "recommended_carnot_use": "Adopt the premise-rule-conclusion record format for future step-level verifier experiments.",
    },
    {
        "rank": 7,
        "title": "Deductive Verification of Chain-of-Thought Reasoning",
        "year": 2023,
        "source": "NeurIPS 2023",
        "category": "stepwise self-verification",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html",
        "why_it_matters": "Still the clearest baseline for decomposing verification into small subprocesses with a constrained natural-language reasoning format.",
        "recommended_carnot_use": "Use as the baseline to beat for stepwise validation and premise-minimization prompts.",
    },
    {
        "rank": 8,
        "title": "Faithful Logical Reasoning via Symbolic Chain-of-Thought",
        "year": 2024,
        "source": "ACL 2024",
        "category": "symbolic CoT translation",
        "url": "https://arxiv.org/abs/2405.18357",
        "why_it_matters": "Translates natural language reasoning into symbolic expressions, then verifies both translation and reasoning with a verifier.",
        "recommended_carnot_use": "Good bridge design for a prompt-answer pair where Carnot first converts text to symbolic state and only then verifies.",
    },
    {
        "rank": 9,
        "title": "Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning",
        "year": 2023,
        "source": "Findings of EMNLP 2023",
        "category": "LLM plus symbolic solver baseline",
        "url": "https://aclanthology.org/2023.findings-emnlp.248/",
        "why_it_matters": "Foundational hybrid pattern: translate to symbolic form, run deterministic inference, and use solver errors to refine the formalization.",
        "recommended_carnot_use": "Keep as the minimum viable pattern for solver-backed extraction experiments.",
    },
    {
        "rank": 10,
        "title": "Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning",
        "year": 2025,
        "source": "arXiv 2510.01069",
        "category": "typed proof view of CoT",
        "url": "https://arxiv.org/abs/2510.01069",
        "why_it_matters": "Provides a formal lens for mapping informal chain-of-thought into typed proof objects and treating well-typedness as a certificate of faithfulness.",
        "recommended_carnot_use": "Use as design guidance for a typed intermediate representation rather than free-form regex over reasoning text.",
    },
]

BENCHMARK_ASSETS: list[dict[str, str]] = [
    {
        "title": "FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models",
        "year": "2024",
        "url": "https://aclanthology.org/2024.acl-long.257/",
        "use_for_carnot": "Seed a prompt-side benchmark with explicit content, situation, style, format, and example constraints.",
    },
    {
        "title": "CFBench: A Comprehensive Constraints-Following Benchmark for LLMs",
        "year": "2025",
        "url": "https://aclanthology.org/2025.acl-long.1581/",
        "use_for_carnot": "Adds broader real-life constraint taxonomies and requirement-priority scoring.",
    },
    {
        "title": "RealInstruct",
        "year": "2024",
        "url": "https://aclanthology.org/2024.findings-emnlp.458/",
        "use_for_carnot": "Real user multi-constraint instructions are a better supervision source than synthetic prompt lists.",
    },
    {
        "title": "VIFBench",
        "year": "2026",
        "url": "https://arxiv.org/abs/2601.17789",
        "use_for_carnot": "Instruction-following verifier benchmark with fine-grained labels; closest external evaluation target to Carnot's gap.",
    },
    {
        "title": "IndusCP",
        "year": "2025",
        "url": "https://aclanthology.org/2025.emnlp-main.809/",
        "use_for_carnot": "Industrial constraint-programming tasks for scheduling and resource-allocation extraction.",
    },
    {
        "title": "P-FOLIO",
        "year": "2024",
        "url": "https://arxiv.org/abs/2410.09207",
        "use_for_carnot": "Human-written stepwise logical proofs for evaluating step-level reasoning extraction.",
    },
    {
        "title": "FormalBench",
        "year": "2025",
        "url": "https://aclanthology.org/2025.acl-long.1068/",
        "use_for_carnot": "Program-semantics benchmark where formal specification inference is the task itself.",
    },
    {
        "title": "StructFlowBench",
        "year": "2025",
        "url": "https://aclanthology.org/2025.findings-acl.486/",
        "use_for_carnot": "Useful if Carnot extends from single-turn extraction to multi-turn structural constraints.",
    },
]

RISK_EVIDENCE: list[dict[str, str]] = [
    {
        "title": "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation",
        "year": "2025",
        "url": "https://arxiv.org/abs/2503.11926",
        "why_it_matters": "Shows chain-of-thought can help oversight, but strong optimization pressure can induce obfuscated reasoning.",
    },
    {
        "title": "Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability",
        "year": "2025",
        "url": "https://arxiv.org/abs/2510.19851",
        "why_it_matters": "Explicit stress test showing some models can hide adversarial intent under obfuscation pressure.",
    },
    {
        "title": "Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity",
        "year": "2025",
        "url": "https://arxiv.org/abs/2510.27378",
        "why_it_matters": "Useful metric design: faithful traces can still be poor monitors when they omit crucial factors.",
    },
    {
        "title": "Diagnosing Pathological Chain-of-Thought in Reasoning Models",
        "year": "2026",
        "url": "https://arxiv.org/abs/2602.13904",
        "why_it_matters": "Gives concrete pathology categories and cheap diagnostics for post-hoc rationalization, encoded reasoning, and internalized reasoning.",
    },
    {
        "title": "Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?",
        "year": "2026",
        "url": "https://arxiv.org/abs/2603.22582",
        "why_it_matters": "Recent cross-model evidence that faithfulness varies sharply by family and hint type, which argues for model-specific monitorability gates.",
    },
]

RANKED_IDEAS: list[dict[str, str]] = [
    {
        "rank": "1",
        "idea": "Prompt-to-constraint intermediate representation with solver fallback",
        "score": "625",
        "why": "NSVIF, DeCRIM, and ConstraintLLM all point to the same fix: extract atomic constraints from the instruction before verifying the answer.",
    },
    {
        "rank": "2",
        "idea": "Benchmark-first extraction workbench",
        "score": "500",
        "why": "FollowBench, CFBench, RealInstruct, and VIFBench provide the missing datasets needed to measure extraction recall and false positives directly.",
    },
    {
        "rank": "3",
        "idea": "Dual-path verification: prompt-answer first, CoT second",
        "score": "500",
        "why": "CoT verification is promising, but monitorability papers say Carnot should never depend on raw CoT alone.",
    },
    {
        "rank": "4",
        "idea": "Typed step-graph verification for arithmetic and logic traces",
        "score": "375",
        "why": "VeriCoT, PCRLLM, Deductive Verification, and Typed CoT all support moving from free-form traces to explicit premises and rules.",
    },
    {
        "rank": "5",
        "idea": "Constraint-programming route for scheduling and resource tasks",
        "score": "240",
        "why": "ConstraintLLM plus IndusCP is the best external path for Carnot's scheduling extractor gap.",
    },
    {
        "rank": "6",
        "idea": "CoT monitorability score and fallback policy",
        "score": "240",
        "why": "Recent monitorability work implies Carnot needs a gate deciding when CoT evidence is safe to trust.",
    },
]

PROPOSED_EXPERIMENTS: list[dict[str, str]] = [
    {
        "id": "EXP-211",
        "title": "Instruction-to-Constraint IR Benchmark",
        "goal": "Build a gold benchmark of atomic prompt constraints from FollowBench, RealInstruct, CFBench, and VIFBench, then measure extraction recall and false positives on instruction-tuned models.",
        "hypothesis": "Prompt-side decomposition will reduce false positives more than answer-only regex extraction because the verifier will know exactly which constraints matter before inspecting the response.",
        "success_criteria": "Atomic constraint recall >= 0.85 on the curated benchmark, satisfied-constraint false-positive rate <= 0.05, and measurable improvement over the current regex plus Z3 promptless path.",
    },
    {
        "id": "EXP-212",
        "title": "Dual-Path CoT Verifier with Typed Step Graphs",
        "goal": "Implement a step-level verifier for arithmetic and logic traces using premise-rule-conclusion records inspired by VeriCoT, PCRLLM, Deductive Verification, and Typed CoT.",
        "hypothesis": "A typed step graph will catch errors that answer-only checking misses, but only when combined with prompt-derived constraints and a fallback to answer-level verification.",
        "success_criteria": "On a live instruction-tuned cohort, catch >= 25% of wrong answers missed by prompt-only verification while adding < 2% extra false positives on correct answers.",
    },
    {
        "id": "EXP-213",
        "title": "CoT Monitorability Audit and Fallback Policy",
        "goal": "Measure whether Qwen and Gemma instruction-tuned models expose enough faithful reasoning to justify CoT-based extraction, using recent faithfulness and pathology metrics.",
        "hypothesis": "Monitorability differs by model family and task, so Carnot should gate CoT extraction behind a measured trust score rather than assuming traces are faithful.",
        "success_criteria": "Produce a per-model monitorability score, a pathology breakdown, and a simple policy that predicts when to trust CoT extraction versus prompt-answer-only verification.",
    },
]

KEY_FINDINGS = [
    "The strongest direct fit is prompt-side instruction verification: convert instructions into atomic constraints first, then verify the answer against them.",
    "Step-level CoT verification is now technically credible, but only when reasoning traces are reformatted into explicit premises, rules, and typed steps.",
    "Benchmark coverage for fine-grained instruction constraints is finally good enough to evaluate extraction quality directly instead of using answer accuracy as a proxy.",
    "Recent monitorability papers make raw chain-of-thought an unsafe sole source of truth; Carnot needs a fallback path that does not trust CoT by default.",
]


def get_scan_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def build_results(scan_timestamp: str) -> dict[str, Any]:
    return {
        "experiment": "Exp 210",
        "title": "Research scan - constraint extraction for instruction-tuned models",
        "scan_date": scan_timestamp,
        "focus_problem": "constraint extraction on instruction-tuned models",
        "queries_run": list(QUERY_THEMES),
        "key_findings": list(KEY_FINDINGS),
        "papers": [dict(item) for item in PAPERS],
        "benchmark_assets": [dict(item) for item in BENCHMARK_ASSETS],
        "risk_evidence": [dict(item) for item in RISK_EVIDENCE],
        "proposed_experiments_2026_04_15": [dict(item) for item in PROPOSED_EXPERIMENTS],
        "recommended_execution_order": ["EXP-211", "EXP-213", "EXP-212"],
        "recommendation_summary": {
            "primary_direction": "Build a prompt-to-constraint intermediate representation before attempting richer answer verification.",
            "secondary_direction": "Treat CoT as optional evidence gated by measured monitorability, not as Carnot's only extraction source.",
        },
    }


def render_references_section() -> str:
    lines = [
        "## 2026-04-12 - Exp 210: Constraint Extraction for Instruction-Tuned Models",
        "",
        "### Core papers",
    ]
    for paper in PAPERS:
        lines.extend(
            [
                f"- **#{paper['rank']} {paper['title']}** ({paper['source']}) - {paper['url']}",
                f"  Why it matters: {paper['why_it_matters']}",
                f"  Carnot use: {paper['recommended_carnot_use']}",
            ]
        )

    lines.extend(["", "### Benchmark assets"])
    for asset in BENCHMARK_ASSETS:
        lines.append(
            f"- **{asset['title']}** ({asset['year']}) - {asset['url']} - {asset['use_for_carnot']}"
        )

    lines.extend(["", "### Monitorability and CoT risk evidence"])
    for risk in RISK_EVIDENCE:
        lines.append(
            f"- **{risk['title']}** ({risk['year']}) - {risk['url']} - {risk['why_it_matters']}"
        )

    lines.append("")
    return "\n".join(lines)


def render_studying_section() -> str:
    lines = [
        "## Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models",
        "",
        "### Ranking update",
        "| Rank | Idea | Score | Why it matters |",
        "|------|------|-------|----------------|",
    ]
    for idea in RANKED_IDEAS:
        lines.append(f"| {idea['rank']} | {idea['idea']} | {idea['score']} | {idea['why']} |")

    lines.extend(["", "### Key takeaways"])
    for finding in KEY_FINDINGS:
        lines.append(f"- {finding}")

    lines.extend(["", "### Proposed experiments for 2026-04-15"])
    for experiment in PROPOSED_EXPERIMENTS:
        lines.extend(
            [
                f"- **{experiment['id']} - {experiment['title']}**",
                f"  Goal: {experiment['goal']}",
                f"  Hypothesis: {experiment['hypothesis']}",
                f"  Success criteria: {experiment['success_criteria']}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def upsert_marked_section(path: Path, start_marker: str, end_marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    block = f"{start_marker}\n{body.rstrip()}\n{end_marker}\n"
    if start_marker in text and end_marker in text:
        prefix, remainder = text.split(start_marker, maxsplit=1)
        _, suffix = remainder.split(end_marker, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + block + suffix.lstrip("\n")
    else:
        updated = text.rstrip() + "\n\n" + block
    path.write_text(updated, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    payload = build_results(get_scan_timestamp())
    upsert_marked_section(
        REFERENCES_PATH,
        REFERENCES_START_MARKER,
        REFERENCES_END_MARKER,
        render_references_section(),
    )
    upsert_marked_section(
        STUDYING_PATH,
        STUDYING_START_MARKER,
        STUDYING_END_MARKER,
        render_studying_section(),
    )
    write_json(RESULTS_PATH, payload)
    print(f"[exp210] wrote {RESULTS_PATH}")
    print(f"[exp210] updated {REFERENCES_PATH}")
    print(f"[exp210] updated {STUDYING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
