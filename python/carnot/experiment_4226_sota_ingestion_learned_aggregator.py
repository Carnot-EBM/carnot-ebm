"""Exp 4226 SOTA ingestion for the learned-aggregator .392 plan.

Spec refs: REQ-REPORT-4226, SCENARIO-REPORT-4226.

This module writes a planning artifact, not a benchmark result. It closes the
`.391 planning sweep` into a concrete SOTA-to-experiment mapping after Exp
4220 trained the ARC verifier and Exp 4221 showed headroom without a
vote-beating flat selector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v392",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_stack_mapping",
        "implication",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_learned_aggregator_mapped_v392"
DEFAULT_FLAGGED_FOR_V392 = "agglm_style_arc_review_reconcile_aggregator_v392"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify "
        "discipline)."
    ),
    "flagged_for_v392": (
        "Closes discover->ingest->plan: names the strongest method for the next "
        "planner (e.g. AggLM-style aggregator for ARC, or an AgentAuditor "
        "localized-evidence verifier)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2509.06870": "https://arxiv.org/abs/2509.06870",
    "2602.09341": "https://arxiv.org/abs/2602.09341",
    "2602.02143": "https://arxiv.org/abs/2602.02143",
    "2603.03417": "https://arxiv.org/abs/2603.03417",
    "2603.03538": "https://arxiv.org/abs/2603.03538",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "AggLM review-and-reconcile solution aggregation",
        "arxiv_id_or_url": "2509.06870",
        "url": "https://arxiv.org/abs/2509.06870",
        "carnot_stack_mapping": (
            "Maps the A2 ARC verifier from a flat reranker into an aggregator: "
            "review the candidate set, reconcile conflicting evidence, and "
            "synthesize or select the final grid only after comparing minority "
            "and majority evidence."
        ),
        "implication": (
            "Exp 4221 had oracle headroom but the learned selector did not beat "
            "vote, so .392 should train an ARC aggregator that can recover "
            "minority-correct answers instead of only assigning independent "
            "candidate scores."
        ),
        "failure_mode": (
            "AggLM is demonstrated on reasoning benchmarks, not ARC grids; its "
            "success does not prove Carnot has enough localized ARC evidence or "
            "a generator that can safely synthesize corrected grids."
        ),
        "experiment_mapping": (
            "Build `agglm_style_arc_review_reconcile_aggregator_v392`: feed the "
            "whole ARC candidate set, vote prior, verifier scores, and local "
            "grid evidence into a review/reconcile aggregator; compare "
            "aggregator@1, flat verifier@1, and vote@1 on wrong-majority tasks."
        ),
    },
    {
        "name": "AgentAuditor localized-evidence reasoning-tree audit",
        "arxiv_id_or_url": "2602.09341",
        "url": "https://arxiv.org/abs/2602.09341",
        "carnot_stack_mapping": (
            "Maps to an evidence audit layer over ARC candidates: localize "
            "where candidate transformations diverge, compare the disputed "
            "regions, and reward evidence-based minority selections over "
            "popular but unsupported answers."
        ),
        "implication": (
            "AgentAuditor gives the efficiency frame for .392: localized "
            "evidence should be compared against both vote and an LLM-judge "
            "audit, not only against flat self-consistency."
        ),
        "failure_mode": (
            "Reasoning trees from multi-agent text traces are richer than the "
            "cached ARC candidate rows; without explicit region-level evidence, "
            "an auditor can collapse back into an expensive judge."
        ),
        "experiment_mapping": (
            "Add a localized-evidence verifier row: identify critical grid "
            "disagreements, score evidence around those regions, and report "
            "accuracy and cost against majority vote and LLM-as-judge."
        ),
    },
    {
        "name": "GenSelect-BoN RL-trained generative selection",
        "arxiv_id_or_url": "2602.02143",
        "url": "https://arxiv.org/abs/2602.02143",
        "carnot_stack_mapping": (
            "Maps to the policy-training recipe for selecting among ARC "
            "candidates: synthesize verified correct/incorrect selection tasks "
            "and train the selector with reinforcement learning rather than "
            "prompt-only ranking."
        ),
        "implication": (
            "A .392 selector can be trained from ARC candidate sets with both "
            "correct and incorrect rows, then evaluated against vote, flat "
            "verifier, and AggLM-style aggregation."
        ),
        "failure_mode": (
            "GenSelect remains a selector. It can choose the best existing "
            "candidate but does not by itself reconcile partial correctness or "
            "repair a grid when all high-frequency candidates are wrong."
        ),
        "experiment_mapping": (
            "Create ARC Best-of-N selection episodes from the Exp 4220 labeled "
            "pool, train a small generative selector with correctness reward, "
            "and measure whether it adds lift beyond the logistic verifier."
        ),
    },
    {
        "name": "MSV cross-candidate multi-sequence verification",
        "arxiv_id_or_url": "2603.03417",
        "url": "https://arxiv.org/abs/2603.03417",
        "carnot_stack_mapping": (
            "Maps to cross-candidate verifier features: score each ARC answer "
            "with awareness of the full candidate set, vote basin, confidence "
            "margin, and competing transformation families."
        ),
        "implication": (
            "The A2 feature set should stop treating rows as isolated examples. "
            "Cross-candidate attention or summary features are necessary for "
            "minority-correct recovery."
        ),
        "failure_mode": (
            "Better calibration across candidates does not create an external "
            "correctness signal by itself; if all cross-candidate features track "
            "frequency, MSV-style scoring can still preserve wrong majorities."
        ),
        "experiment_mapping": (
            "Add a candidate-set encoder or explicit cross-candidate summaries "
            "to the ARC verifier, then ablate isolated scoring versus whole-set "
            "scoring on the `wrong_majority_n=5` support slice."
        ),
    },
    {
        "name": "SR-TTRL and online CoT-verifier self-learning loop",
        "arxiv_id_or_url": "2603.03538",
        "url": "https://arxiv.org/abs/2603.03538",
        "carnot_stack_mapping": (
            "Maps to the Phase-B verifier-as-reward loop: use a verifier to "
            "produce higher-fidelity pseudo-labels than majority vote, while "
            "tracking the soundness/completeness trade-off of online verifier "
            "feedback."
        ),
        "implication": (
            "Only after the .392 aggregator shows positive selection value "
            "should Carnot let the verifier supervise self-learning; the reward "
            "loop needs abstention and asymmetric error budgets."
        ),
        "failure_mode": (
            "Self-learning can amplify false verifier positives. The CoT theory "
            "paper studies proof-like traces, and SR-TTRL is an ICML track item, "
            "so neither source removes the need for ARC-specific controls."
        ),
        "experiment_mapping": (
            "Gate a verifier-as-reward self-learning run behind a positive "
            "aggregator result, then compare verifier labels, majority labels, "
            "and random-label controls with explicit soundness and completeness "
            "accounting."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-15: learned aggregator map for .392

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_learned_aggregator_mapped_v392`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `AggLM review-and-reconcile solution aggregation`, arxiv_id_or_url: `2509.06870`, url: `https://arxiv.org/abs/2509.06870`}
  - {name: `AgentAuditor localized-evidence reasoning-tree audit`, arxiv_id_or_url: `2602.09341`, url: `https://arxiv.org/abs/2602.09341`}
  - {name: `GenSelect-BoN RL-trained generative selection`, arxiv_id_or_url: `2602.02143`, url: `https://arxiv.org/abs/2602.02143`}
  - {name: `MSV cross-candidate multi-sequence verification`, arxiv_id_or_url: `2603.03417`, url: `https://arxiv.org/abs/2603.03417`}
  - {name: `SR-TTRL and online CoT-verifier self-learning loop`, arxiv_id_or_url: `2603.03538`, url: `https://arxiv.org/abs/2603.03538`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v392: `agglm_style_arc_review_reconcile_aggregator_v392`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. AggLM-style aggregator for ARC, or an AgentAuditor localized-evidence verifier).

## Fresh-pass provenance

Read `research-references.md` `.391 planning sweep`, `research-studying.md`,
`results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json`, and
`results/experiment_4221_oracle_distinct_arc_verifier_beats_vote.json`.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "AggLM solution aggregation AgentAuditor majority vote LLM judge GenSelect Best-of-N multi-sequence verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "self reflective test time reinforcement learning chain of thought verifier online learnability" --limit 8`

The cluster helper emitted the broadened verifier/process-reward and
energy/verifier arXiv API URLs. Semantic Scholar returned HTTP 429 for both
focused queries, so no S2-only promotion is claimed. Low-concurrency
WebSearch/WebFetch verified arXiv:2509.06870, arXiv:2602.09341,
arXiv:2602.02143, arXiv:2603.03417, and arXiv:2603.03538. The SR-TTRL ICML
listing was also checked as the self-reflective pseudo-labeling companion to
the CoT-verifier theory anchor.

## Exp 4220 A2 status and Exp 4221 A3 status

Exp 4220 did train an oracle-distinct ARC verifier:
`selector_trained=true`, `oracle_distinct_auroc=0.778980279`, CI95
`[0.6146676853, 0.9174508427]`, `verifier_is_oracle=false`, and
`wrong_majority_n=5`. The sparse-positive warning remains load-bearing:
only 14 positive candidates were available out of 1796 in the stratified set.

Exp 4221 then ran the A3 gate and found headroom without a vote-beating
selector: `oracle_at_k=1.0`, `oracle_minus_vote=0.3571428571`,
`verifier_minus_vote_delta=-0.0714285714`, CI95 `[-0.2142857143, 0.0]`, and
`oracle_distinct_beats_vote=false`. This is a complete ingestion target, not a
green selector result: the next method must recover wrong-majority answers
that flat reranking missed.

## SOTA -> experiment mapping

## Review-and-reconcile aggregation

**Method/source:** AggLM, arXiv:2509.06870
(https://arxiv.org/abs/2509.06870), trains aggregation as an explicit
reasoning skill: review candidates, reconcile disagreements, and synthesize a
final answer. AgentAuditor, arXiv:2602.09341
(https://arxiv.org/abs/2602.09341), audits localized branch evidence and beats
both majority vote and LLM-as-judge.

**Carnot stack mapping:** Strengthen the A2 ARC verifier into an aggregator.
The aggregator should see the whole candidate set, vote prior, verifier scores,
localized grid disagreements, and any partial-correctness evidence. It should
review, reconcile, and synthesize or select, not merely rerank candidates by an
independent logistic score.

**Implication:** The .392 headline should target the wrong-majority slice from
Exp 4220/4221: correct answer present, vote wrong or insufficient, and flat
verifier not enough. AggLM is the closest precedent because it explicitly
recovers minority-correct answers; AgentAuditor supplies the localized evidence
and LLM-judge efficiency frame.

**Failure mode:** These sources do not prove ARC aggregation. AggLM is not an
ARC grid system, and AgentAuditor assumes reasoning-tree evidence that cached
ARC rows may not contain. Without region-level evidence, aggregation can become
an expensive judge or an overfit selector.

**Experiment mapping:** Flag
`agglm_style_arc_review_reconcile_aggregator_v392`. Build an ARC aggregator
that reviews all candidates and either synthesizes a final grid or chooses a
candidate after localized reconciliation. Compare aggregator@1 against vote@1,
flat verifier@1, conservative override, and LLM-as-judge on wrong-majority
tasks with bootstrap CI and matched cost.

## RL-trained generative selection

**Method/source:** GenSelect-BoN, arXiv:2602.02143
(https://arxiv.org/abs/2602.02143), trains small models with DAPO on generated
Best-of-N selection tasks and reports selection gains over prompting and
majority-voting baselines.

**Carnot stack mapping:** Convert Exp 4220 rows into ARC Best-of-N selection
episodes with verified correct and incorrect candidates, then train a
generative selector as the selection-only baseline against the aggregator.

**Implication:** A learned selector can still be useful, but it should be
treated as the selection recipe, not the full reconciliation recipe. It answers
whether RL selection improves on the logistic verifier before synthesis is
introduced.

**Failure mode:** GenSelect cannot recover an answer that is only partially
present across multiple candidates, and it can still overvalue popular
wrong-majority clusters if the reward design does not isolate minority-correct
cases.

**Experiment mapping:** Build ARC Best-of-N selection episodes, train a DAPO-like
selector on correct/incorrect candidates, and compare it with the AggLM-style
aggregator and the Exp 4221 flat verifier.

## Cross-candidate verification

**Method/source:** MSV, arXiv:2603.03417
(https://arxiv.org/abs/2603.03417), jointly processes multiple candidate
solutions and models interactions across them instead of scoring each candidate
in isolation.

**Carnot stack mapping:** Add candidate-set context to the ARC verifier:
vote_weight, self_consistency_margin, verifier margins, basin features,
localized disagreement summaries, and cross-candidate calibration should be
available in one scoring pass.

**Implication:** The Exp 4220 feature list already points this way with
cross-candidate self-consistency and per-cell confidence. .392 should make that
explicit and ablate isolated scoring against whole-set scoring.

**Failure mode:** Calibration over a candidate set can still track frequency
rather than truth. MSV-style context must be paired with oracle-distinct local
evidence, or it can preserve the wrong majority more confidently.

**Experiment mapping:** Add an MSV-style candidate-set encoder or explicit
whole-set summaries to A2, then report isolated verifier, cross-candidate
verifier, and review/reconcile aggregator on the same wrong-majority tasks.

## Self-learning verifier-as-reward loop

**Method/source:** SR-TTRL, checked through the ICML 2026 listing, frames
self-reflective verification as a way to create higher-fidelity pseudo-labels
than majority vote. Online Learnability of Chain-of-Thought Verifiers,
arXiv:2603.03538 (https://arxiv.org/abs/2603.03538), provides the
soundness/completeness lens for verifier feedback loops.

**Carnot stack mapping:** This is Phase B, not the .392 headline: use a
positive aggregator/verifier as reward only after it has shown external
selection value, and track false-accept versus false-reject errors explicitly.

**Implication:** If the AggLM-style aggregator clears the wrong-majority gate,
Carnot can try self-learning with verifier pseudo-labels. If it does not, vote
pseudo-labels and verifier labels are both unsafe training targets.

**Failure mode:** Self-training can amplify verifier mistakes. Soundness errors
are worse than ordinary selection misses because they poison the generator or
aggregator that will later create more traces.

**Experiment mapping:** Gate a verifier-as-reward self-learning run behind a
positive aggregator result, then compare verifier pseudo-labels, majority
pseudo-labels, and random-label controls under explicit soundness/completeness
accounting.

## Flagged for .392

`agglm_style_arc_review_reconcile_aggregator_v392` is the strongest single
method for the next planner. The reason is direct: Exp 4221 already showed
headroom but no flat selector lift, and AggLM is the closest verified precedent
for recovering minority-correct answers by review and reconciliation. Use
AgentAuditor as the localized-evidence and LLM-judge efficiency comparator, but
run the AggLM-style ARC aggregator before another flat rerank.
"""

STUDYING_SECTION = """## 2026-06-15 Exp 4226 - .391 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-learned-aggregator-v392-2026-06-15.md`.

**Filtered track:** learned aggregation after Exp 4220 trained the ARC verifier
and Exp 4221 found oracle headroom but `oracle_distinct_beats_vote=false`.

**Seed and fresh-pass candidates marked ingested:**
- AggLM, arXiv:2509.06870 - mapped as the strongest .392 follow-up: convert
  the A2 ARC verifier from flat rerank into review/reconcile/synthesize
  aggregation for minority-correct recovery.
- AgentAuditor, arXiv:2602.09341 - mapped to localized evidence auditing and
  the LLM-as-judge efficiency head-to-head.
- GenSelect-BoN, arXiv:2602.02143 - mapped as the RL-trained selection-only
  baseline and recipe.
- MSV, arXiv:2603.03417 - mapped to cross-candidate features and whole-set
  verifier calibration.
- Online CoT-verifier learnability, arXiv:2603.03538, plus SR-TTRL ICML 2026 -
  mapped to the verifier-as-reward self-learning loop after a positive
  aggregator gate.

Exp 4220 status mapped honestly: `selector_trained=true`,
`oracle_distinct_auroc=0.778980279`, and `wrong_majority_n=5`. Exp 4221 status
mapped honestly: `oracle_minus_vote=0.3571428571`,
`verifier_minus_vote_delta=-0.0714285714`, and
`oracle_distinct_beats_vote=false`.

flagged_for_v392:
`agglm_style_arc_review_reconcile_aggregator_v392`.

Flagged for .392: `agglm_style_arc_review_reconcile_aggregator_v392`.

**Bottom line for the .392 roadmap:** run the AggLM-style ARC aggregator before another flat rerank.
"""

STUDYING_MARKER = "## 2026-06-15 Exp 4226 - .391 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v392: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4226 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v392": flagged_for_v392,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so uncited method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 5 or len(methods_mapped) > 8:
        raise ValueError("methods_mapped must contain five to eight methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_stack_mapping, implication, failure_mode, and "
                "experiment_mapping"
            )
        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method arxiv_id_or_url must be a verified source: {source}")
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        expected_url = VERIFIED_SOURCE_URLS[source]
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v392"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v392 must be a non-empty string")
    if flagged != DEFAULT_FLAGGED_FOR_V392:
        raise ValueError("flagged_for_v392 must name the AggLM-style ARC aggregator")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "Exp 4220 A2 status",
        "Exp 4221 A3 status",
        "SOTA -> experiment mapping",
        "Review-and-reconcile aggregation",
        "RL-trained generative selection",
        "Cross-candidate verification",
        "Self-learning verifier-as-reward loop",
        "review",
        "reconcile",
        "synthesize",
        "localized evidence",
        "DAPO",
        "Best-of-N",
        "multi-sequence",
        "cross-candidate",
        "soundness",
        "completeness",
        "verifier-as-reward",
        "Carnot stack mapping",
        "Implication",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .392",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")
    if "oracle_distinct_beats_vote=false" not in markdown:
        raise ValueError("markdown note must preserve oracle_distinct_beats_vote=false")
    if "wrong_majority_n=5" not in markdown:
        raise ValueError("markdown note must preserve wrong_majority_n=5")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v392=DEFAULT_FLAGGED_FOR_V392,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def _replace_or_insert_section(
    existing: str,
    *,
    marker: str,
    section: str,
) -> str:
    if marker not in existing:
        if existing.startswith("## "):
            return section + "\n" + existing
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + section
        return existing.replace("\n## ", "\n" + section + "\n## ", 1)

    before, after_marker = existing.split(marker, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + section.rstrip() + "\n"
    return before + section + after_marker[next_section + 1 :]


def _with_studying_section(existing: str) -> str:
    return _replace_or_insert_section(
        existing,
        marker=STUDYING_MARKER,
        section=STUDYING_SECTION,
    )


def main() -> int:
    """Write the default Exp 4226 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-learned-aggregator-v392-2026-06-15.md",
        artifact_path=repo_root
        / "results/experiment_4226_sota_ingestion_learned_aggregator.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
