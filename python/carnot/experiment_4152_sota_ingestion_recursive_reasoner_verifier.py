"""Exp 4152 SOTA ingestion for recursive reasoners and energy guidance.

Spec refs: REQ-REPORT-4152, SCENARIO-REPORT-4152.

This module writes a planning artifact, not a benchmark result. The core risk
is mixing three separate claims: a recursive generator can improve, a verifier
can learn from accepted and rejected traces, and a diffusion sampler can accept
external guidance during generation. The validators below keep those claims
citation-backed and force the .385 handoff to name one gated next method rather
than treating DiffusionGemma, SEDD, or classifier guidance as already-measured
Carnot verifier evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


DIFFUSIONGEMMA_URL = "https://ai.google.dev/gemma/docs/diffusiongemma"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v385",
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
DEFAULT_HONEST_VERDICT = (
    "complete: sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped"
)
DEFAULT_FLAGGED_FOR_V385 = "diffusiongemma_sedd_verifier_energy_guidance_probe_v385"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method/source MUST carry a real arXiv ID/URL; an ingestion note "
        "without verifiable citations is treated as fabrication."
    ),
    "flagged_for_v385": (
        "Closes the discover->ingest->plan loop: names the strongest method "
        "for the next planner."
    ),
}

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2510.04871",
        "2511.02886",
        "2402.06457",
        "2310.16834",
        "2105.05233",
        "2207.12598",
    }
)
VERIFIED_CANONICAL_URLS = frozenset({DIFFUSIONGEMMA_URL})
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    {
        "arXiv:2510.04871",
        "arXiv:2511.02886",
        "arXiv:2402.06457",
        "arXiv:2310.16834",
        "arXiv:2105.05233",
        "arXiv:2207.12598",
        DIFFUSIONGEMMA_URL,
    }
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "TRM nano-trm recursive baseline gate",
        "arxiv_id_or_url": "2510.04871",
        "url": "https://arxiv.org/abs/2510.04871",
        "implementation_over_stack": (
            "Keep nano-trm as the recursive Sudoku substrate and measure oracle "
            "headroom before attributing gains to the Carnot verifier."
        ),
        "failure_mode": (
            "An undertrained or no-headroom baseline makes verifier-guided training "
            "and energy guidance uninformative."
        ),
    },
    {
        "name": "TTA-TRM adaptation-control arm",
        "arxiv_id_or_url": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "implementation_over_stack": (
            "Run the same bounded fine-tuning budget without Carnot verifier labels "
            "so adaptation compute is isolated from verifier value."
        ),
        "failure_mode": (
            "Full fine-tuning can improve the tiny model by itself, so a "
            "verifier-labeled arm without this control overclaims causality."
        ),
    },
    {
        "name": "V-STaR accepted/rejected trace selector",
        "arxiv_id_or_url": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "implementation_over_stack": (
            "Retain accepted and rejected nano-trm traces and train a selector or "
            "pairwise verifier before spending on another generator pass."
        ),
        "failure_mode": (
            "If the candidate pool has correlated errors or false-positive labels, "
            "the selector learns artifacts rather than correctness."
        ),
    },
    {
        "name": "SEDD discrete diffusion score-energy formalism",
        "arxiv_id_or_url": "2310.16834",
        "url": "https://arxiv.org/abs/2310.16834",
        "implementation_over_stack": (
            "Use score-entropy discrete diffusion as the formal bridge for adding "
            "Carnot verifier energy during denoising instead of after it."
        ),
        "failure_mode": (
            "SEDD is a generator objective, not a verifier; an uncalibrated external "
            "energy can damage fluency or collapse diversity."
        ),
    },
    {
        "name": "Classifier-guided diffusion energy precedent",
        "arxiv_id_or_url": "2105.05233",
        "url": "https://arxiv.org/abs/2105.05233",
        "implementation_over_stack": (
            "Treat Carnot verifier scores as the discrete-token analogue of a "
            "guidance energy that reshapes the denoising choice distribution."
        ),
        "failure_mode": (
            "Over-guidance can trade away diversity and create verifier-shaped but "
            "invalid samples unless guidance weights are ablated."
        ),
    },
    {
        "name": "Classifier-free diffusion guidance control",
        "arxiv_id_or_url": "2207.12598",
        "url": "https://arxiv.org/abs/2207.12598",
        "implementation_over_stack": (
            "Keep a no-external-verifier guidance control so Carnot energy is "
            "compared against ordinary conditional/unconditional score mixing."
        ),
        "failure_mode": (
            "A guidance win can come from generic conditioning strength rather than "
            "the Carnot verifier unless this control is included."
        ),
    },
    {
        "name": "DiffusionGemma queued discrete-text substrate",
        "arxiv_id_or_url": DIFFUSIONGEMMA_URL,
        "url": DIFFUSIONGEMMA_URL,
        "implementation_over_stack": (
            "Queue DiffusionGemma as the open-weight block-diffusion generator for "
            "verifier-energy guidance after the verifier discrimination gate."
        ),
        "failure_mode": (
            "DiffusionGemma is a generator substrate, not evidence that Carnot "
            "verifier guidance works; base-task quality and the gate must be measured."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: recursive reasoner verifier energy-guidance map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_recursive_reasoner_verifier_energy_guidance_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM nano-trm recursive baseline gate`, arxiv_id_or_url: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM adaptation-control arm`, arxiv_id_or_url: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `SEDD discrete diffusion score-energy formalism`, arxiv_id_or_url: `2310.16834`, url: `https://arxiv.org/abs/2310.16834`}
  - {name: `Classifier-guided diffusion energy precedent`, arxiv_id_or_url: `2105.05233`, url: `https://arxiv.org/abs/2105.05233`}
  - {name: `Classifier-free diffusion guidance control`, arxiv_id_or_url: `2207.12598`, url: `https://arxiv.org/abs/2207.12598`}
  - {name: `DiffusionGemma queued discrete-text substrate`, arxiv_id_or_url: `https://ai.google.dev/gemma/docs/diffusiongemma`, url: `https://ai.google.dev/gemma/docs/diffusiongemma`}
  - principle: Each method/source MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v385: `diffusiongemma_sedd_verifier_energy_guidance_probe_v385`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read `research-studying.md` and `research-references.md` filtered to
verifier-guided-training and energy-guided-generation, including the prior
Exp 4102, 4111, 4121, 4130, 4141 entries and the 2026-06-13
DiffusionGemma operator-requested note. Ran the reliable helpers, not
`/deep-research`:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training TRM V-STaR recursive reasoning" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "energy guided generation discrete diffusion classifier guidance SEDD" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "DiffusionGemma energy guidance discrete token diffusion verifier" --limit 8`

The arXiv cluster helper emitted the broadened verifier, EBM, and
active-inference query URLs. Semantic Scholar returned HTTP 429 for the first
two focused queries and returned `arXiv:2605.04040` for the DiffusionGemma
guidance query; that paper is adjacent evidence for verification feedback as
generation guidance, but the `.385` map keeps the requested TRM/TTA/V-STaR/
SEDD/guidance anchors as the load-bearing sources. Low-concurrency
WebSearch/WebFetch verified `arXiv:2510.04871`, `arXiv:2511.02886`,
`arXiv:2402.06457`, `arXiv:2310.16834`, `arXiv:2105.05233`,
`arXiv:2207.12598`, and `https://ai.google.dev/gemma/docs/diffusiongemma`.

## Current .385 verifier-guided-generation anchor

The local DiffusionGemma note says the model is a DEPTH scale-up of the
verifier-as-guidance bet, not a new proof that the verifier works. The same
note keeps it gated on the TRM verifier graft reporting `verifier_value_added
== true`. Therefore the `.385` handoff should connect two tracks without
collapsing them: recursive `nano-trm` still measures candidate quality and
verifier discrimination, while DiffusionGemma/SEDD supplies the generation-time
surface where Carnot energy could act before final text or grid selection.

## TRM nano-trm recursive baseline gate

**Method/source:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the baseline recursive substrate: a
small two-layer recursive model with 7M parameters and strong Sudoku/ARC
generalization claims relative to larger systems.

**Implementation over nano-trm + Carnot-verifier stack:** Keep `nano-trm` as
the local Sudoku baseline that measures oracle(best-of-K) headroom,
pass-at-one, checkpoint lineage, and candidate diversity before any
energy-guided generation claim. The verifier is only meaningful when the
generator emits alternatives the verifier can discriminate.

**Pitfalls / where it fails:** If the baseline is undertrained, lacks oracle
headroom, or majority vote already captures all selectable support, then a
verifier-training or diffusion-guidance result is uninformative rather than
negative evidence against the method.

## TTA-TRM adaptation-control arm

**Method/source:** TTA-TRM, `arXiv:2511.02886`
(https://arxiv.org/abs/2511.02886), shows that bounded full fine-tuning of a
tiny recursive model can change results within a competition-style compute
budget.

**Implementation over nano-trm + Carnot-verifier stack:** Keep a no-verifier
adaptation arm with the same checkpoint, optimizer-step budget, LR schedule,
candidate pool, and receipts as any Carnot-verifier-labeled arm. This is the
control that prevents ordinary adaptation compute from being mistaken for
verifier reward.

**Pitfalls / where it fails:** Full fine-tuning can win through compute alone.
If the verifier arm gets more steps, cleaner labels, or a different schedule,
the experiment cannot attribute a delta to Carnot verifier information.

## V-STaR accepted/rejected trace selector

**Method/source:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to select among candidates at
inference time.

**Implementation over nano-trm + Carnot-verifier stack:** Retain accepted and
rejected `nano-trm` Sudoku traces, then train or evaluate a selector over that
paired evidence before spending on another generator pass. The clean handoff is
a pairwise selector that can be compared against fixed vote, oracle(best-of-K),
and the current Carnot verifier order.

**Pitfalls / where it fails:** V-STaR needs real contrast. If the saved pool is
dominated by near-identical wrong candidates or the executable oracle exposes
false-positive Carnot labels, the selector will learn trace artifacts instead
of correctness.

## SEDD discrete diffusion score-energy formalism

**Method/source:** SEDD, `arXiv:2310.16834`
(https://arxiv.org/abs/2310.16834), extends score matching to discrete spaces
through score entropy and provides the clearest language-model bridge from
token denoising to score/energy reasoning.

**Implementation over nano-trm + Carnot-verifier stack:** Use SEDD as the
formal scaffold for the queued DiffusionGemma guidance experiment: Carnot
verifier energy should alter denoising choices while the canvas is still
mutable, then the resulting candidate is scored by the same executable and
non-oracle receipts used for `nano-trm`. This moves verifier signal upstream
from rerank-after-generation to generation-time guidance.

**Pitfalls / where it fails:** SEDD itself is a generator loss, not a Carnot
verifier. If the external energy is badly scaled or applied too late, it can
collapse diversity, harm fluency, or merely reproduce post-hoc reranking under
a more expensive sampler.

## Classifier-guided diffusion energy precedent

**Method/source:** Classifier-guided diffusion, `arXiv:2105.05233`
(https://arxiv.org/abs/2105.05233), shows that diffusion samples can be steered
by an external classifier gradient. Classifier-free guidance,
`arXiv:2207.12598` (https://arxiv.org/abs/2207.12598), is the matching control:
guidance can also be produced by mixing model scores without an external
classifier.

**Implementation over nano-trm + Carnot-verifier stack:** Treat Carnot verifier
scores as the discrete-token analogue of an external guidance energy. The
`.385` probe should sweep small guidance weights, include a no-external-energy
classifier-free-style control, and report both generation quality and exact
Sudoku validity so an over-guided sample cannot pass as a verifier win.

**Pitfalls / where it fails:** Guidance is a tradeoff. Too much verifier energy
can reduce diversity or create samples optimized for a proxy rather than for
validity. A win against no-guidance is not enough unless it also beats the
ordinary conditional/unconditional guidance control.

## DiffusionGemma queued discrete-text substrate

**Method/source:** DiffusionGemma official documentation
(https://ai.google.dev/gemma/docs/diffusiongemma) describes an experimental
open model that generates text with discrete diffusion over block canvases,
including parallel denoising, bidirectional attention over the generation
canvas, entropy-bounded denoising, adaptive stopping, and a Sudoku fine-tuning
recipe.

**Implementation over nano-trm + Carnot-verifier stack:** Queue DiffusionGemma
as the open-weight substrate for the `.385` guidance probe, not as a replacement
headline. The testable path is: establish verifier discrimination on the
`nano-trm` domain, attach Carnot energy during DiffusionGemma denoising, and
compare against no-guidance plus classifier-free-style guidance controls.

**Pitfalls / where it fails:** DiffusionGemma is a generator substrate. Its base
or SFT Sudoku behavior does not prove Carnot verifier value, and the official
docs describe it as experimental. The result must be labeled as guidance only
if Carnot energy changes the denoising outcome and improves held-out exact
validity under matched compute.

## Flagged for the .385 roadmap

`diffusiongemma_sedd_verifier_energy_guidance_probe_v385` is the strongest
candidate. It directly tests the queued energy-guided-generation hypothesis:
SEDD gives the discrete score/energy formalism, classifier/classifier-free
guidance gives the ablation structure, and DiffusionGemma gives the open
block-diffusion substrate. Keep it gated on measured Carnot-verifier
discrimination; otherwise spend `.385` on improving the trace selector and
candidate diversity before launching a guided-generation probe.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4152 - .385 recursive-reasoner/verifier energy-guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-energy-guidance-2026-06-13.md`.

**Filtered track:** verifier-guided training plus energy-guided generation for
the `.385` handoff. This connects the TRM/TTA/V-STaR recursive verifier stack
to the queued DiffusionGemma energy-guidance use without treating a generator
substrate as verifier evidence.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the `nano-trm` baseline and oracle-headroom
  gate before any verifier-guided or diffusion-guided claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget adaptation-control arm
  that prevents full fine-tuning from masquerading as verifier reward.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector for
  saved `nano-trm` candidates before another generator pass.
- SEDD, arXiv:2310.16834 - mapped as the discrete diffusion score/energy
  formalism for generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free diffusion
  guidance, arXiv:2207.12598 - mapped as the external-guidance precedent and
  no-external-verifier control.
- DiffusionGemma official docs, https://ai.google.dev/gemma/docs/diffusiongemma
  - mapped as the queued open-weight block-diffusion substrate, gated on
  measured Carnot-verifier discrimination.

Flagged for .385: `diffusiongemma_sedd_verifier_energy_guidance_probe_v385`.

**Bottom line for the .385 roadmap:** run the DiffusionGemma/SEDD
verifier-energy-guidance probe only if the verifier discrimination gate is
positive; otherwise keep improving the V-STaR-style trace selector and
candidate diversity before spending on guided-generation probe.
"""

STUDYING_MARKER = (
    "## 2026-06-13 Exp 4152 - .385 recursive-reasoner/verifier energy-guidance "
    "SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v385: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4152 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v385": flagged_for_v385,
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

    flagged = artifact["flagged_for_v385"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v385 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps the verified sources to .385 work."""

    required_phrases = (
        "Current .385 verifier-guided-generation anchor",
        "TRM nano-trm recursive baseline gate",
        "TTA-TRM adaptation-control arm",
        "V-STaR accepted/rejected trace selector",
        "SEDD discrete diffusion score-energy formalism",
        "Classifier-guided diffusion energy precedent",
        "DiffusionGemma queued discrete-text substrate",
        "Implementation over nano-trm + Carnot-verifier stack",
        "Pitfalls / where it fails",
        "Flagged for the .385 roadmap",
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
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying-section update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v385=DEFAULT_FLAGGED_FOR_V385,
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
        / "docs/research-notes/"
        "sota-ingestion-recursive-reasoner-verifier-energy-guidance-2026-06-13.md",
        artifact_path=repo_root
        / "results/experiment_4152_sota_ingestion_recursive_reasoner_verifier.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
