"""Exp 4141 SOTA ingestion for recursive reasoners and verifier rewards.

Spec refs: REQ-REPORT-4141, SCENARIO-REPORT-4141.

This module writes a planning artifact, not a benchmark result. The important
risk is citation drift: a headline like "GRAM is the next generator" or
"Weaver is the right reranker" is only useful if the next planner can trace it
to a real paper and can see the condition under which it should be acted on.
The validators below keep the JSON schema small, force every method row to use
verified citations, and make the .384 GRAM flag conditional on verifier value
instead of turning a stronger generator into a fake verifier win.
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
        "flagged_for_v384",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "implementation_over_stack",
        "failure_mode",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_recursive_reasoner_verifier_mapped"
DEFAULT_FLAGGED_FOR_V384 = (
    "gram_as_generator_if_verifier_value_added_and_headroom_present_v384"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method/source MUST carry a real arXiv ID or canonical doc URL; "
        "an ingestion note without verifiable citations is treated as fabrication."
    ),
    "flagged_for_v384": (
        "Closes the discover->ingest->plan loop: names the strongest method "
        "for the next planner (candidate: GRAM-as-generator IF verifier_value_added)."
    ),
}

VERIFIED_ARXIV_IDS = frozenset({"2605.19376", "2602.08498", "2506.18203"})
VERIFIED_CANONICAL_URLS = frozenset[str]()
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    {"arXiv:2605.19376", "arXiv:2602.08498", "arXiv:2506.18203"}
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "GRAM stochastic-latent generator",
        "arxiv_id_or_url": "2605.19376",
        "url": "https://arxiv.org/abs/2605.19376",
        "implementation_over_stack": (
            "Treat GRAM as the .384 generator candidate only after the TRM graft "
            "shows non-oracle verifier value and best-of-K headroom."
        ),
        "failure_mode": (
            "A stronger generator can erase rerank headroom; without verifier_value_added "
            "the graft would only benchmark GRAM rather than Carnot verifier value."
        ),
    },
    {
        "name": "TRM thinking reward for RLVR/GRPO",
        "arxiv_id_or_url": "2602.08498",
        "url": "https://arxiv.org/abs/2602.08498",
        "implementation_over_stack": (
            "Use verified-correct trace filtering as the precedent for the .383 "
            "RFT A-vs-B de-confound: verifier-certified labels versus vote labels."
        ),
        "failure_mode": (
            "If correctness filtering is mixed with label-source effects, the run "
            "cannot distinguish verifier reward from generic adaptation compute."
        ),
    },
    {
        "name": "Weaver weak-verifier weighted ensemble",
        "arxiv_id_or_url": "2506.18203",
        "url": "https://arxiv.org/abs/2506.18203",
        "implementation_over_stack": (
            "Make the .383 non-oracle ensemble-rerank headline a weighted weak-verifier "
            "baseline rather than a single executable-oracle rerank."
        ),
        "failure_mode": (
            "Weak-verifier weights can overfit correlated errors; oracle(best-of-K) must "
            "beat vote before any null rerank is interpreted."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: recursive reasoner generator plus verifier-as-reward map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_recursive_reasoner_verifier_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `GRAM stochastic-latent generator`, arxiv_id_or_url: `2605.19376`, url: `https://arxiv.org/abs/2605.19376`}
  - {name: `TRM thinking reward for RLVR/GRPO`, arxiv_id_or_url: `2602.08498`, url: `https://arxiv.org/abs/2602.08498`}
  - {name: `Weaver weak-verifier weighted ensemble`, arxiv_id_or_url: `2506.18203`, url: `https://arxiv.org/abs/2506.18203`}
  - principle: Each method/source MUST carry a real arXiv ID or canonical doc URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v384: `gram_as_generator_if_verifier_value_added_and_headroom_present_v384`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner (candidate: GRAM-as-generator IF verifier_value_added).

**Fresh-pass provenance**

Read the 2026-06-13 Post-.382 Planning Sweep in `research-references.md` and
the recursive-reasoner / verifier-as-reward track in `research-studying.md`,
including the Exp 4081, 4102, 4111, 4121, and 4130 ingestions. Ran the reliable
helpers, not `/deep-research`:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Generative Recursive Reasoning Models GRAM Sudoku Extreme stochastic latent" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Thinking Reward Model TRM GRPO RLVR verified-correct traces" --limit 8`

The arXiv cluster helper emitted the broadened verifier, EBM, and
active-inference query URLs. Semantic Scholar returned HTTP 429 for both
focused queries, so it did not displace the primary-paper anchors. Low-volume
WebSearch/WebFetch verified `arXiv:2605.19376`, `arXiv:2602.08498`, and
`arXiv:2506.18203`.

## Current .383 recursive-reasoner plus verifier anchor

Exp 4139 is the current graft receipt: `verifier_value_added=false`,
`headroom_present=false`, `graft_deferred=true`, and the honest verdict is
`complete: uninformative_no_headroom_false_negative_risk`. That means the next
planner should not treat the no-lift result as evidence against verifier
reward. It should also not jump to GRAM as a headline replacement unless the
next run first creates measurable oracle headroom and then shows that the
non-oracle verifier or the RFT label contrast captures some of it.

## GRAM stochastic-latent generator

**Method/source:** GRAM, `arXiv:2605.19376`
(https://arxiv.org/abs/2605.19376), turns deterministic recursive refinement
into probabilistic multi-trajectory latent computation. The paper reports
97.0% Sudoku-Extreme test accuracy with a 10M-parameter model, above TRM's
87.4% and HRM's 55.0% in the same table.

**Implementation over nano-trm + Carnot-verifier stack:** Treat GRAM as the
strongest .384 generator candidate only if the verifier side earns the right
to consume a stronger candidate distribution. The concrete graft is:
GRAM samples multiple latent trajectories per Sudoku puzzle, the executable
oracle is used only to measure best-of-K headroom, and the Carnot non-oracle
energy/text-stat ensemble tries to recover that headroom without seeing the
exact-validity label.

**Pitfalls / where it fails:** GRAM can reduce or erase the reranker headroom
that Carnot needs to measure verifier value. If `verifier_value_added` remains
false or `headroom_present` remains false, a GRAM run is a generator benchmark,
not a verifier-as-reward result.

## TRM thinking reward for RLVR/GRPO

**Method/source:** Characterizing, Evaluating, and Optimizing Complex
Reasoning, `arXiv:2602.08498` (https://arxiv.org/abs/2602.08498), trains a
Thinking Reward Model from verified-correct reasoning traces and integrates it
as an auxiliary thinking reward inside RLVR/GRPO. The key precedent for Carnot
is that reasoning-quality shaping is isolated from answer correctness by
filtering to verified-correct traces first.

**Implementation over nano-trm + Carnot-verifier stack:** This supports the
.383 RFT de-confound. Arm A should use verifier-certified labels, arm B should
use vote-certified labels, and both arms should share the same baseline
checkpoint, candidate pool, optimizer budget, and scheduler receipts. The
measured claim is not "training improved"; it is whether the verifier label
source adds held-out value beyond a vote label source under the same adaptation
compute.

**Pitfalls / where it fails:** If the pipeline mixes correctness filtering,
candidate diversity, and label-source effects, the result cannot isolate
verifier reward. A positive delta would be uninterpretable if arm B lacks the
same adaptation budget, and a null is uninformative if the candidate pool has
no best-of-K headroom.

## Weaver weak-verifier weighted ensemble

**Method/source:** Weaver, `arXiv:2506.18203`
(https://arxiv.org/abs/2506.18203), combines multiple weak verifiers with
weak-supervision-derived weights. Its repeated-sampling setting is directly
aligned with Carnot's candidate-pool rerank question: generated candidates are
scored, normalized, and selected by a combined verifier score rather than by a
single weak verifier or unweighted vote.

**Implementation over nano-trm + Carnot-verifier stack:** Use Weaver as the
peer baseline for the .383 non-oracle ensemble-rerank headline. The executable
Sudoku checker remains an oracle upper bound, not the transferable result.
The transferable result is whether weighted continuous Sudoku energy plus
text-stat/verifier features beats fixed vote when oracle(best-of-K) proves
there is selectable headroom.

**Pitfalls / where it fails:** Weak-verifier weighting assumes enough diversity
that errors are not fully correlated. If all weak features track the same
near-valid dead ends, weighting can amplify the wrong candidate. The mandatory
positive control is still oracle(best-of-K) versus vote before interpreting a
non-oracle null.

## Flagged for the .384 roadmap

`gram_as_generator_if_verifier_value_added_and_headroom_present_v384` is the
strongest .384 candidate. GRAM is the best next generator because its
stochastic-latent trajectories naturally produce a distribution for Carnot to
rerank, but it should be scheduled only behind a positive headroom/value gate,
not as an unconditional rerank claim.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4141 - .383 recursive-reasoner/verifier SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-2026-06-13.md`.

**Filtered track:** recursive reasoner generator choice plus verifier-as-reward
mapping for the `.383` decisive graft. This follows the Exp 4130 resumable
training ingestion and the Exp 4139 graft receipt, which currently reports
`verifier_value_added=false`, `headroom_present=false`, and
`complete: uninformative_no_headroom_false_negative_risk`.

**Seed and fresh-pass candidates marked ingested:**
- GRAM, arXiv:2605.19376 - mapped as the stochastic-latent generator to graft
  onto in `.384` only if a verifier-value/headroom gate is met.
- Thinking Reward Model for complex reasoning, arXiv:2602.08498 - mapped as
  the RLVR/GRPO precedent for isolating verified-correct trace quality from
  outcome correctness, directly informing the `.383` RFT de-confound.
- Weaver, arXiv:2506.18203 - mapped as the weighted weak-verifier ensemble
  precedent for the `.383` non-oracle ensemble-rerank headline.

Flagged for .384: `gram_as_generator_if_verifier_value_added_and_headroom_present_v384`.

**Bottom line for the .384 roadmap:** use GRAM as the next generator only if
the verifier side first demonstrates transferable value with measurable
oracle(best-of-K) headroom; otherwise continue fixing headroom/candidate
diversity, not as an unconditional rerank claim.
"""

STUDYING_MARKER = (
    "## 2026-06-13 Exp 4141 - .383 recursive-reasoner/verifier SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v384: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4141 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v384": flagged_for_v384,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the exact JSON contract so uncited method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    field_principles = artifact["field_principles"]
    if field_principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 3:
        raise ValueError("methods_mapped must contain at least three methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "implementation_over_stack, and failure_mode"
            )
        source = method["arxiv_id_or_url"]
        if source in VERIFIED_ARXIV_IDS:
            expected_url = f"https://arxiv.org/abs/{source}"
        elif source in VERIFIED_CANONICAL_URLS:
            expected_url = source
        else:
            raise ValueError(
                "method arxiv_id_or_url must be a verified arxiv ID or canonical URL: "
                f"{source}"
            )
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v384"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v384 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps the three source anchors to .384 work."""

    required_phrases = (
        "Current .383 recursive-reasoner plus verifier anchor",
        "GRAM stochastic-latent generator",
        "TRM thinking reward for RLVR/GRPO",
        "Weaver weak-verifier weighted ensemble",
        "Implementation over nano-trm + Carnot-verifier stack",
        "Pitfalls / where it fails",
        "Flagged for the .384 roadmap",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying-section update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v384=DEFAULT_FLAGGED_FOR_V384,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing = studying_path.read_text(encoding="utf-8")
    studying_path.write_text(_with_studying_section(existing), encoding="utf-8")
    return artifact


def _with_studying_section(existing: str) -> str:
    if STUDYING_MARKER not in existing:
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + STUDYING_SECTION
        return existing.replace("\n## ", "\n" + STUDYING_SECTION + "\n## ", 1)

    before, after_marker = existing.split(STUDYING_MARKER, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + STUDYING_SECTION.rstrip() + "\n"
    return before + STUDYING_SECTION + after_marker[next_section + 1 :]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-recursive-reasoner-verifier-2026-06-13.md",
        artifact_path=repo_root
        / "results/experiment_4141_sota_ingestion_recursive_reasoner_verifier.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
