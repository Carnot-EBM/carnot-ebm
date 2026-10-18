"""Exp 4162 SOTA ingestion for verifier moat and diffusion guidance.

Spec refs: REQ-REPORT-4162, SCENARIO-REPORT-4162.

This module writes a planning artifact, not a benchmark result. It keeps the
.385 verifier moat separate from the queued DiffusionGemma generator work:
external verifier evidence must first beat or complement self-consistency under
matched compute, and only then should the next planner spend on reward guidance
during discrete diffusion denoising.
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
        "flagged_for_v386",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_moat_implication",
        "efficiency_implication",
        "diffusiongemma_guidance_implication",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_verifier_moat_guidance_mapped"
DEFAULT_FLAGGED_FOR_V386 = (
    "entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method/source MUST carry a real arXiv ID/URL (verified); "
        "an ingestion note without verifiable citations is treated as fabrication."
    ),
    "flagged_for_v386": (
        "Closes the discover->ingest->plan loop: names the strongest method "
        "for the next planner."
    ),
}

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2605.26172",
        "2504.16828",
        "2510.13918",
        "2505.04842",
        "2602.05000",
        "2605.05138",
        "2603.24621",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{arxiv_id}" for arxiv_id in VERIFIED_ARXIV_IDS
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "ARBITER reasoning-basin verifier-moat anchor",
        "arxiv_id_or_url": "2605.26172",
        "url": "https://arxiv.org/abs/2605.26172",
        "carnot_moat_implication": (
            "Majority vote can choose the largest wrong basin, so Carnot should "
            "measure external verifier recovery as an additive vote-orthogonal "
            "rerank signal, not as a vote replacement."
        ),
        "efficiency_implication": (
            "Keep the cheap verifier between fixed vote and LLM-judge: recover "
            "oracle headroom only when it improves accuracy per unit compute."
        ),
        "diffusiongemma_guidance_implication": (
            "Only guide DiffusionGemma after the verifier distinguishes correct "
            "minority basins from stable wrong basins on the current stack."
        ),
    },
    {
        "name": "ThinkPRM data-efficient process verifier",
        "arxiv_id_or_url": "2504.16828",
        "url": "https://arxiv.org/abs/2504.16828",
        "carnot_moat_implication": (
            "Process verification can beat self-consistency when it checks the "
            "reasoning path; Carnot should compare verifier-plus-vote selection "
            "against vote and judge baselines."
        ),
        "efficiency_implication": (
            "ThinkPRM sets the LLM-judge comparison bar: verifier compute must "
            "scale better than asking a large judge to rescore every candidate."
        ),
        "diffusiongemma_guidance_implication": (
            "Use process-style scores as intermediate guidance signals rather "
            "than waiting until the denoised candidate is complete."
        ),
    },
    {
        "name": "Optimal LLM+PRM aggregation",
        "arxiv_id_or_url": "2510.13918",
        "url": "https://arxiv.org/abs/2510.13918",
        "carnot_moat_implication": (
            "The verifier should be calibrated as a weighted aggregation term "
            "with vote evidence; replacing the vote can throw away useful LLM "
            "prior information."
        ),
        "efficiency_implication": (
            "Precompute aggregation weights so the verifier improves test-time "
            "scaling without multiplying candidate-generation cost."
        ),
        "diffusiongemma_guidance_implication": (
            "Expose guidance weights as an ablation knob for mixing base "
            "denoising confidence with Carnot verifier energy."
        ),
    },
    {
        "name": "RLV unified reasoner-verifier value head",
        "arxiv_id_or_url": "2505.04842",
        "url": "https://arxiv.org/abs/2505.04842",
        "carnot_moat_implication": (
            "Training a verifier/value capability alongside reasoning supports "
            "an external-value rerank arm, but the moat claim still requires a "
            "vote-plus-verifier head-to-head."
        ),
        "efficiency_implication": (
            "The next efficiency gate should compare a cheap verifier/value head "
            "against LLM-judge rescoring under matched parallel sampling."
        ),
        "diffusiongemma_guidance_implication": (
            "A learned value head is a plausible reward source for guidance, but "
            "it must be checked against executable verifier labels before use."
        ),
    },
    {
        "name": "EntRGi entropy-aware reward guidance",
        "arxiv_id_or_url": "2602.05000",
        "url": "https://arxiv.org/abs/2602.05000",
        "carnot_moat_implication": (
            "EntRGi is not moat evidence by itself; it becomes relevant only "
            "after Carnot proves the external reward/verifier is discriminative."
        ),
        "efficiency_implication": (
            "Guidance can spend verifier calls during denoising, so the .386 "
            "gate must report reward-call cost versus post-hoc judge rescoring."
        ),
        "diffusiongemma_guidance_implication": (
            "Use entropy-aware interpolation between soft token relaxations and "
            "hard tokens as the template for Carnot energy over DiffusionGemma."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id_or_url": "2605.05138",
        "url": "https://arxiv.org/abs/2605.05138",
        "carnot_moat_implication": (
            "Executable world models make verifier-grounded transitions the "
            "selection primitive; the moat is action recovery, not just answer "
            "selection."
        ),
        "efficiency_implication": (
            "Use RHAE/action efficiency as the cost axis when verifier-pruned "
            "planning replaces brute-force exploration."
        ),
        "diffusiongemma_guidance_implication": (
            "Treat generated world-model edits as candidates that can be guided "
            "or pruned by executable transition energy before acting."
        ),
    },
    {
        "name": "ARC-AGI-3 technical report",
        "arxiv_id_or_url": "2603.24621",
        "url": "https://arxiv.org/abs/2603.24621",
        "carnot_moat_implication": (
            "ARC-AGI-3 makes adaptive efficiency the benchmark target, so the "
            "verifier moat must improve actions-to-progress under real rules."
        ),
        "efficiency_implication": (
            "Report human-action-normalized efficiency and avoid claims that "
            "only improve raw solve count by spending more actions."
        ),
        "diffusiongemma_guidance_implication": (
            "Guided generation should target compact executable hypotheses and "
            "plans, not only fluent natural-language reasoning traces."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: verifier moat and DiffusionGemma guidance map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_moat_guidance_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARBITER reasoning-basin verifier-moat anchor`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `ThinkPRM data-efficient process verifier`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `Optimal LLM+PRM aggregation`, arxiv_id_or_url: `2510.13918`, url: `https://arxiv.org/abs/2510.13918`}
  - {name: `RLV unified reasoner-verifier value head`, arxiv_id_or_url: `2505.04842`, url: `https://arxiv.org/abs/2505.04842`}
  - {name: `EntRGi entropy-aware reward guidance`, arxiv_id_or_url: `2602.05000`, url: `https://arxiv.org/abs/2602.05000`}
  - {name: `Executable World Models for ARC-AGI-3`, arxiv_id_or_url: `2605.05138`, url: `https://arxiv.org/abs/2605.05138`}
  - {name: `ARC-AGI-3 technical report`, arxiv_id_or_url: `2603.24621`, url: `https://arxiv.org/abs/2603.24621`}
  - principle: Each method/source MUST carry a real arXiv ID/URL (verified); an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v386: `entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-studying.md` and `research-references.md` filtered to
verifier-vs-self-consistency, reward-guided-generation, and ARC-AGI-3, plus
`results/experiment_4152_sota_ingestion_recursive_reasoner_verifier.json` for
the DiffusionGemma gate provenance. The prior TRM/TTA-TRM/V-STaR/SEDD/CFG
milestone ingestion is treated as already banked and is not duplicated here.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_clusters.py 3 --max-results 8`
- `python3 scripts/sweep_semscholar.py "verifier self consistency process reward aggregation LLM PRM" --limit 8`
- `python3 scripts/sweep_semscholar.py "reward guided generation diffusion language model energy guidance" --limit 8`
- `python3 scripts/sweep_semscholar.py "ARC-AGI-3 executable world models tech report" --limit 8`

The cluster helper emitted the broadened verifier, energy, and world-model
arXiv API URLs. Semantic Scholar returned `2510.13918` among eight IDs for the
verifier/self-consistency query and HTTP 429 for the reward-guidance and
ARC-AGI-3 queries. Low-concurrency WebSearch/WebFetch verified all seven
requested arXiv anchors: arXiv:2605.26172, arXiv:2504.16828, arXiv:2510.13918,
arXiv:2505.04842, arXiv:2602.05000, arXiv:2605.05138, and arXiv:2603.24621.

## SOTA -> experiment mapping

## ARBITER reasoning-basin verifier-moat anchor

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), shows sampled reasoning trajectories
cluster into basins and that majority vote can pick a stable wrong basin even
when the correct answer is present in the candidate pool.

**Carnot moat implication:** This is the .385 rerank-recovery design anchor:
the external Carnot verifier must recover correct minority candidates that
self-consistency misses. The verifier should aggregate with the vote, because
vote mass is still useful evidence; it should not replace the vote blindly.

**Efficiency implication:** The relevant metric is recovered oracle headroom per
unit cost. A cheap executable verifier earns its place only if it recovers
wrong-majority cases at lower cost than LLM-judge rescoring.

**DiffusionGemma guidance implication:** Do not launch guidance just because a
diffusion substrate exists. First require a positive discrimination gate showing
the verifier can separate correct minority basins from stable wrong basins.

## ThinkPRM data-efficient process verifier

**Method/source:** Process Reward Models That Think, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), trains generative process verifiers with far
fewer process labels than ordinary discriminative PRMs and reports wins against
LLM-as-judge and other verifier baselines.

**Carnot moat implication:** It is the positive existence proof that a verifier
can beat self-consistency when it checks the process instead of only final
answer agreement. Carnot should score verifier-plus-vote against vote-only,
LLM-judge, and post-hoc verifier-only baselines.

**Efficiency implication:** ThinkPRM makes the LLM-judge efficiency comparison
load-bearing: if a verifier is more expensive than judge rescoring, it is not
the Carnot moat.

**DiffusionGemma guidance implication:** Process-style partial scores are the
right shape for denoising-time guidance, where the sampler needs intermediate
signals before a final candidate exists.

## Optimal LLM+PRM aggregation

**Method/source:** Optimal Aggregation of LLM and PRM Signals for Efficient
Test-Time Scaling, arXiv:2510.13918 (https://arxiv.org/abs/2510.13918), argues
for calibrated weighted aggregation of LLM and PRM signals and reports better
test-time scaling efficiency than vanilla weighted vote.

**Carnot moat implication:** The verifier should be a calibrated term in the
selector, not an unconditional replacement for the vote. The .385 artifact
should therefore report vote-only, verifier-only, and calibrated
vote-plus-verifier arms.

**Efficiency implication:** Precomputed aggregation weights are the cheap path:
spend compute once to calibrate the selector rather than repeatedly increasing
candidate K or sending each candidate to a large judge.

**DiffusionGemma guidance implication:** The same calibration lesson applies to
guidance weights. Carnot energy should be swept and mixed with the base
denoising confidence, with no-guidance and base-guidance controls.

## RLV unified reasoner-verifier value head

**Method/source:** Putting the Value Back in RL, arXiv:2505.04842
(https://arxiv.org/abs/2505.04842), co-trains LLM reasoners with a generative
verifier/value capability and reports efficient parallel test-time scaling.

**Carnot moat implication:** RLV supports the reward-graft thesis: a value or
verifier head can make parallel samples more selectable. The Carnot claim still
needs the external-verifier-vs-self-consistency head-to-head and a
vote-plus-verifier aggregation arm.

**Efficiency implication:** The strongest .386 efficiency test is a cheap
verifier/value head versus LLM-judge rescoring under matched candidate pools
and matched parallel sampling.

**DiffusionGemma guidance implication:** A learned value head is a plausible
reward for token-level or trace-level guidance, but only after executable
labels confirm it tracks validity rather than model preference artifacts.

## EntRGi entropy-aware reward guidance

**Method/source:** EntRGi, arXiv:2602.05000
(https://arxiv.org/abs/2602.05000), studies reward guidance for discrete
diffusion language models by interpolating between continuous token relaxations
and hard tokens according to predictive entropy.

**Carnot moat implication:** EntRGi is not evidence that Carnot has a moat. It
is the implementation template to use after the moat gate is positive.

**Efficiency implication:** Guidance spends verifier/reward calls during
denoising rather than after generation. The next run must report cost per
reward call and compare it to post-hoc rerank and LLM-judge baselines.

**DiffusionGemma guidance implication:** This is the strongest .386 method:
apply Carnot verifier energy through entropy-aware soft/hard token
interpolation during DiffusionGemma denoising, gated on positive verifier
discrimination.

## Executable World Models for ARC-AGI-3

**Method/source:** Executable World Models for ARC-AGI-3, arXiv:2605.05138
(https://arxiv.org/abs/2605.05138), evaluates coding agents that maintain,
verify, refactor, and plan through executable Python world models for ARC-AGI-3.

**Carnot moat implication:** The moat can be transition verification and action
selection, not only answer reranking. Carnot should keep executable validation
as the external signal that self-consistency lacks.

**Efficiency implication:** Use action efficiency and RHAE-style metrics. A
verifier that reduces actions-to-progress is valuable even before it increases
the headline solve count.

**DiffusionGemma guidance implication:** Guided generation can target compact
world-model edits, transition hypotheses, and plans. The verifier energy should
favor executable hypotheses that survive held-out transition checks.

## ARC-AGI-3 technical report

**Method/source:** ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence,
arXiv:2603.24621 (https://arxiv.org/abs/2603.24621), defines an interactive
benchmark centered on exploration, goal inference, world-model building, and
human-action-normalized adaptive efficiency.

**Carnot moat implication:** The verifier moat must show up as better adaptive
progress under real rules, not only as a better static answer selector.

**Efficiency implication:** The official benchmark framing makes efficiency
load-bearing. Report actions, RHAE-style ratios, and solve progress rather than
letting more compute masquerade as better intelligence.

**DiffusionGemma guidance implication:** Diffusion guidance should be aimed at
executable hypotheses and action plans that reduce exploration cost, not just
more fluent reasoning text.

## Flagged for .386

`entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`
is the strongest follow-on. It should run only after .385 verifier
discrimination is positive. If the verifier does not beat or complement
self-consistency under calibrated vote aggregation, the next planner should
choose the RLV-style energy-verifier-vs-LLM-judge efficiency head-to-head
instead of spending on DiffusionGemma guidance.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4162 - .386 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md`.

**Filtered track:** verifier-vs-self-consistency, reward-guided generation,
and ARC-AGI-3 action efficiency for the `.386` handoff. This ingestion extends
the `.385` verifier moat and queued DiffusionGemma gate without duplicating the
prior TRM/TTA-TRM/V-STaR/SEDD/CFG milestone ingestion.

**Seed and fresh-pass candidates marked ingested:**
- ARBITER, arXiv:2605.26172 - mapped as the wrong-majority/rerank-recovery
  moat anchor and the reason to aggregate an external verifier with vote.
- ThinkPRM, arXiv:2504.16828 - mapped as the data-efficient process-verifier
  existence proof and LLM-judge comparison bar.
- Optimal LLM+PRM Aggregation, arXiv:2510.13918 - mapped as the calibrated
  vote-plus-verifier aggregation recipe.
- RLV, arXiv:2505.04842 - mapped as the cheap verifier/value-head efficiency
  head-to-head template.
- EntRGi, arXiv:2602.05000 - mapped as the discrete diffusion reward-guidance
  template for DiffusionGemma after a positive discrimination gate.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, and ARC-AGI-3 tech
  report, arXiv:2603.24621 - mapped as executable transition verification and
  action-efficiency anchors.

Flagged for .386: `entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`.

**Bottom line for the .386 roadmap:** run the EntRGi-style DiffusionGemma
energy-guidance template only after the verifier-discrimination gate is
positive. If the gate is not positive, run the RLV-style cheap
energy-verifier-vs-LLM-judge efficiency head-to-head first.
"""

REFERENCES_SECTION = """## 2026-06-13 Exp 4162 - verifier-moat guidance ingestion note

**Status:** INGESTED. Full note:
`docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md`.
Artifact: `results/experiment_4162_sota_ingestion_verifier_moat_guidance.json`.

**Scope:** .385 verifier moat plus .386 DiffusionGemma guidance gate. This
section intentionally does not duplicate the prior TRM/TTA-TRM/V-STaR/SEDD/CFG
ingestion; those remain banked in Exp 4152.

**Verified sources and mapping summary:**
- ARBITER, arXiv:2605.26172, https://arxiv.org/abs/2605.26172 - wrong-majority
  basins justify an external verifier that aggregates with vote rather than
  replacing it.
- ThinkPRM, arXiv:2504.16828, https://arxiv.org/abs/2504.16828 - process
  verification is the data-efficient alternative to LLM-judge rescoring.
- Optimal LLM+PRM Aggregation, arXiv:2510.13918,
  https://arxiv.org/abs/2510.13918 - calibrate verifier scores with vote
  evidence instead of selecting by raw PRM score alone.
- RLV, arXiv:2505.04842, https://arxiv.org/abs/2505.04842 - use a cheap
  verifier/value head as the efficiency head-to-head against LLM judges.
- EntRGi, arXiv:2602.05000, https://arxiv.org/abs/2602.05000 - strongest
  DiffusionGemma guidance template after positive verifier discrimination.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138,
  https://arxiv.org/abs/2605.05138 - executable transition verification maps
  the moat to action recovery.
- ARC-AGI-3 technical report, arXiv:2603.24621,
  https://arxiv.org/abs/2603.24621 - action efficiency and RHAE-style scoring
  define the ARC-AGI-3 north-star metric.

flagged_for_v386:
`entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`.
"""

STUDYING_MARKER = (
    "## 2026-06-13 Exp 4162 - .386 verifier-moat guidance SOTA ingestion ingested"
)
REFERENCES_MARKER = (
    "## 2026-06-13 Exp 4162 - verifier-moat guidance ingestion note"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v386: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4162 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v386": flagged_for_v386,
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
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 5:
        raise ValueError("methods_mapped must contain at least five methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_moat_implication, efficiency_implication, and "
                "diffusiongemma_guidance_implication"
            )
        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_ARXIV_IDS:
            raise ValueError(f"method arxiv_id_or_url must be a verified arxiv ID: {source}")
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        expected_url = f"https://arxiv.org/abs/{source}"
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    flagged = artifact["flagged_for_v386"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v386 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to the three axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "ARBITER reasoning-basin verifier-moat anchor",
        "ThinkPRM data-efficient process verifier",
        "Optimal LLM+PRM aggregation",
        "RLV unified reasoner-verifier value head",
        "EntRGi entropy-aware reward guidance",
        "Executable World Models for ARC-AGI-3",
        "ARC-AGI-3 technical report",
        "Carnot moat implication",
        "Efficiency implication",
        "DiffusionGemma guidance implication",
        "Flagged for .386",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(
            f"markdown note missing verified source citations: {missing_sources}"
        )


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
    references_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent research-file updates."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v386=DEFAULT_FLAGGED_FOR_V386,
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
    references_path.write_text(
        _with_references_section(references_path.read_text(encoding="utf-8")),
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


def _with_references_section(existing: str) -> str:
    return _replace_or_insert_section(
        existing,
        marker=REFERENCES_MARKER,
        section=REFERENCES_SECTION,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md",
        artifact_path=repo_root
        / "results/experiment_4162_sota_ingestion_verifier_moat_guidance.json",
        studying_path=repo_root / "research-studying.md",
        references_path=repo_root / "research-references.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
