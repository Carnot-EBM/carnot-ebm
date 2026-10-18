"""Exp 4170 SOTA ingestion for verifier guidance into the .387 handoff.

Spec refs: REQ-REPORT-4170, SCENARIO-REPORT-4170.

This module writes a planning artifact, not a benchmark result. The current
decision is deliberately conservative: Exp 4168 deferred the verifier graft
because the TRM baseline was not faithful and stable, so DiffusionGemma remains
queued behind a positive verifier-discrimination gate instead of being treated
as an earned next step.
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
        "flagged_for_v387",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_verifier_implication",
        "queued_diffusiongemma_implication",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = (
    "complete: sota_ingestion_verifier_moat_guidance_mapped_v387"
)
DEFAULT_FLAGGED_FOR_V387 = (
    "vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387"
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method/source MUST carry a real arXiv ID/URL; an ingestion note "
        "without verifiable citations is treated as fabrication."
    ),
    "flagged_for_v387": (
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
        "2602.05000",
        "2410.21357",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{arxiv_id}" for arxiv_id in VERIFIED_ARXIV_IDS
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "TRM nano-trm baseline and headroom gate",
        "arxiv_id_or_url": "2510.04871",
        "url": "https://arxiv.org/abs/2510.04871",
        "carnot_verifier_implication": (
            "Use nano-trm as the candidate generator only after the baseline is "
            "faithful and has oracle headroom; a verifier cannot add value when "
            "the pool has no selectable correct alternatives."
        ),
        "queued_diffusiongemma_implication": (
            "DiffusionGemma should inherit the same headroom gate: do not spend "
            "on denoising-time verifier energy until the TRM stack emits diverse "
            "candidates the verifier can rank."
        ),
        "experiment_mapping": (
            "For .387, keep the stable checkpoint and candidate-diversity receipts "
            "as preconditions before any verifier graft or diffusion guidance run."
        ),
    },
    {
        "name": "TTA-TRM adaptation-control arm",
        "arxiv_id_or_url": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "carnot_verifier_implication": (
            "Any verifier-labeled improvement must beat a same-budget full "
            "fine-tuning control, because tiny recursive models can improve from "
            "adaptation compute alone."
        ),
        "queued_diffusiongemma_implication": (
            "A DiffusionGemma guidance arm needs a no-verifier adaptation or "
            "conditioning control so ordinary generator adaptation is not counted "
            "as external verifier value."
        ),
        "experiment_mapping": (
            "Carry a matched no-verifier arm with identical checkpoint, candidate "
            "pool, optimizer-step accounting, and wall-clock budget."
        ),
    },
    {
        "name": "V-STaR accepted/rejected trace selector",
        "arxiv_id_or_url": "2402.06457",
        "url": "https://arxiv.org/abs/2402.06457",
        "carnot_verifier_implication": (
            "Keep both accepted and rejected traces so the selector learns a "
            "correctness boundary rather than only imitating successful outputs."
        ),
        "queued_diffusiongemma_implication": (
            "The same paired traces can calibrate verifier energy before it is "
            "attached to DiffusionGemma; without rejected evidence, guidance can "
            "optimize style artifacts."
        ),
        "experiment_mapping": (
            "Flag this as .387: build the rejected-trace selector and headroom "
            "gate first, then unlock EntRGi-style guidance only if discrimination "
            "turns positive."
        ),
    },
    {
        "name": "SEDD discrete diffusion score-energy formalism",
        "arxiv_id_or_url": "2310.16834",
        "url": "https://arxiv.org/abs/2310.16834",
        "carnot_verifier_implication": (
            "SEDD is not verifier evidence; it is the discrete score and energy "
            "language needed to place Carnot scores inside a token denoising loop."
        ),
        "queued_diffusiongemma_implication": (
            "Use SEDD as the formal bridge for applying verifier energy while the "
            "DiffusionGemma canvas is still mutable, with exact checks after the "
            "candidate is committed."
        ),
        "experiment_mapping": (
            "When the verifier gate is positive, sweep small energy weights and "
            "report exact validity, diversity, and reward-call cost."
        ),
    },
    {
        "name": "Classifier-guided diffusion external-energy precedent",
        "arxiv_id_or_url": "2105.05233",
        "url": "https://arxiv.org/abs/2105.05233",
        "carnot_verifier_implication": (
            "Treat Carnot verifier scores as an external guidance signal, but "
            "audit for proxy over-optimization because guidance trades diversity "
            "against fidelity."
        ),
        "queued_diffusiongemma_implication": (
            "A DiffusionGemma probe should expose guidance strength as an ablation "
            "knob and refuse a win that only makes samples verifier-shaped."
        ),
        "experiment_mapping": (
            "Include no-guidance, weak-guidance, and strong-guidance arms with "
            "matched denoising steps and exact downstream validation."
        ),
    },
    {
        "name": "Classifier-free guidance control",
        "arxiv_id_or_url": "2207.12598",
        "url": "https://arxiv.org/abs/2207.12598",
        "carnot_verifier_implication": (
            "A generic model-score guidance control is required so external "
            "verifier value is not confused with ordinary score mixing."
        ),
        "queued_diffusiongemma_implication": (
            "DiffusionGemma should compare Carnot-verifier energy against a "
            "classifier-free-style internal guidance control before claiming a moat."
        ),
        "experiment_mapping": (
            "Make the control arm mandatory in the future guidance experiment, "
            "not an optional appendix."
        ),
    },
    {
        "name": "EntRGi entropy-aware reward guidance",
        "arxiv_id_or_url": "2602.05000",
        "url": "https://arxiv.org/abs/2602.05000",
        "carnot_verifier_implication": (
            "EntRGi is the best concrete guidance mechanism once the verifier is "
            "known to be discriminative, but it does not itself prove the verifier "
            "has value."
        ),
        "queued_diffusiongemma_implication": (
            "Use entropy-aware interpolation between soft token relaxations and "
            "hard tokens as the DiffusionGemma implementation template after the "
            "gate flips positive."
        ),
        "experiment_mapping": (
            "Keep EntRGi queued behind the .387 V-STaR/headroom gate rather than "
            "launching it from the current deferred graft state."
        ),
    },
    {
        "name": "EDLM sequence-level diffusion energy comparator",
        "arxiv_id_or_url": "2410.21357",
        "url": "https://arxiv.org/abs/2410.21357",
        "carnot_verifier_implication": (
            "EDLM supplies an internal sequence-energy comparator; Carnot should "
            "measure whether an external executable verifier adds information "
            "beyond diffusion-model energy."
        ),
        "queued_diffusiongemma_implication": (
            "For DiffusionGemma, compare external Carnot energy with an internal "
            "sequence-level energy baseline so guidance is not just an EBM "
            "approximation claim."
        ),
        "experiment_mapping": (
            "Use EDLM as the future internal-energy control once the external "
            "verifier discrimination gate is satisfied."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: verifier-moat guidance map for .387

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_moat_guidance_mapped_v387`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM nano-trm baseline and headroom gate`, arxiv_id_or_url: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM adaptation-control arm`, arxiv_id_or_url: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id_or_url: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `SEDD discrete diffusion score-energy formalism`, arxiv_id_or_url: `2310.16834`, url: `https://arxiv.org/abs/2310.16834`}
  - {name: `Classifier-guided diffusion external-energy precedent`, arxiv_id_or_url: `2105.05233`, url: `https://arxiv.org/abs/2105.05233`}
  - {name: `Classifier-free guidance control`, arxiv_id_or_url: `2207.12598`, url: `https://arxiv.org/abs/2207.12598`}
  - {name: `EntRGi entropy-aware reward guidance`, arxiv_id_or_url: `2602.05000`, url: `https://arxiv.org/abs/2602.05000`}
  - {name: `EDLM sequence-level diffusion energy comparator`, arxiv_id_or_url: `2410.21357`, url: `https://arxiv.org/abs/2410.21357`}
  - principle: Each method/source MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v387: `vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-studying.md` and `research-references.md` filtered to
verifier-as-reward, verifier-guided trace selection, and energy-guided
generation. Also checked the latest gate artifact,
`results/experiment_4168_decisive_verifier_graft_defensive.json`: it records
`verifier_value_added=false` because the graft was deferred while the baseline
was not faithful/stable, not because DiffusionGemma guidance was tested.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier as reward recursive reasoning V-STaR TRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "energy guided generation discrete diffusion language model classifier guidance SEDD" --limit 8`

The cluster helper emitted the broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for both focused queries during this run.
Low-concurrency WebSearch/WebFetch then verified the mapped paper set:
arXiv:2510.04871, arXiv:2511.02886, arXiv:2402.06457, arXiv:2310.16834,
arXiv:2105.05233, arXiv:2207.12598, arXiv:2602.05000, and arXiv:2410.21357.
The DiffusionGemma official documentation was also checked as queued substrate
context, but it is not counted as evidence that the verifier works.

## SOTA -> experiment mapping

## TRM nano-trm baseline and headroom gate

**Method/source:** TRM, arXiv:2510.04871
(https://arxiv.org/abs/2510.04871), is the tiny recursive baseline: a 7M
parameter two-layer recursive model with strong puzzle generalization claims.

**Carnot-verifier implication:** The verifier is only meaningful when the TRM
candidate pool is faithful, stable, and has oracle headroom. If the generator
does not emit selectable correct alternatives, reranking and reward guidance
are uninformative.

**Queued DiffusionGemma implication:** DiffusionGemma should inherit the same
headroom gate. A faster or deeper denoising substrate cannot fix an unmeasured
verifier signal.

**Experiment mapping:** For .387, preserve checkpoint lineage, candidate
diversity, oracle(best-of-K), and vote baselines before any verifier graft or
diffusion guidance run.

## TTA-TRM adaptation-control arm

**Method/source:** TTA-TRM, arXiv:2511.02886
(https://arxiv.org/abs/2511.02886), shows that bounded full fine-tuning of a
tiny recursive model can change ARC outcomes inside a competition-style budget.

**Carnot-verifier implication:** A verifier-labeled arm needs a same-budget
no-verifier adaptation arm. Otherwise ordinary adaptation compute can masquerade
as verifier reward.

**Queued DiffusionGemma implication:** A guidance result must include a
no-external-verifier adaptation or conditioning control, because a generator can
improve from its own update path without Carnot information.

**Experiment mapping:** Keep identical checkpoint, optimizer-step, LR schedule,
candidate-pool, and wall-clock receipts across verifier and no-verifier arms.

## V-STaR accepted/rejected trace selector

**Method/source:** V-STaR, arXiv:2402.06457
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to select among candidates.

**Carnot-verifier implication:** This is the strongest .387 method because it
acts on the current bottleneck: Carnot needs paired accepted/rejected traces and
a selector before spending on a generator-side guidance stack.

**Queued DiffusionGemma implication:** The same rejected evidence should
calibrate Carnot energy before it is attached to DiffusionGemma. Without
rejected traces, guidance can optimize superficial trace artifacts.

**Experiment mapping:** Build the rejected-trace selector and headroom gate
first. Unlock EntRGi-style DiffusionGemma guidance only if verifier
discrimination turns positive.

## SEDD discrete diffusion score-energy formalism

**Method/source:** SEDD, arXiv:2310.16834
(https://arxiv.org/abs/2310.16834), extends score matching to discrete spaces
through score entropy and provides the bridge from token denoising to
score/energy reasoning.

**Carnot-verifier implication:** SEDD is not verifier evidence. It is the
formal scaffold for moving Carnot scores upstream from post-hoc reranking into
generation-time energy.

**Queued DiffusionGemma implication:** Apply verifier energy while the
DiffusionGemma canvas remains mutable, then check the committed candidate with
the same exact validation receipts.

**Experiment mapping:** After a positive discrimination gate, sweep small
guidance weights and report exact validity, diversity, and reward-call cost.

## Classifier-guided diffusion external-energy precedent

**Method/source:** Classifier-guided diffusion, arXiv:2105.05233
(https://arxiv.org/abs/2105.05233), demonstrates steering diffusion samples
with an external classifier signal and shows the fidelity/diversity tradeoff.

**Carnot-verifier implication:** Carnot verifier scores can play the role of an
external guidance signal, but the experiment must audit proxy over-optimization.

**Queued DiffusionGemma implication:** The DiffusionGemma probe should expose
guidance strength and refuse to count verifier-shaped but invalid samples as a
win.

**Experiment mapping:** Include no-guidance, weak-guidance, and strong-guidance
arms under matched denoising steps and exact downstream validation.

## Classifier-free guidance control

**Method/source:** Classifier-free guidance, arXiv:2207.12598
(https://arxiv.org/abs/2207.12598), mixes conditional and unconditional model
scores to obtain guidance without an external classifier.

**Carnot-verifier implication:** This is the mandatory no-external-verifier
control: it distinguishes generic score mixing from actual Carnot verifier
value.

**Queued DiffusionGemma implication:** DiffusionGemma guidance must beat or
complement this internal-score control before making a verifier-moat claim.

**Experiment mapping:** Treat classifier-free-style control as a required arm
in any future guidance experiment.

## EntRGi entropy-aware reward guidance

**Method/source:** EntRGi, arXiv:2602.05000
(https://arxiv.org/abs/2602.05000), studies reward guidance for discrete
diffusion language models by interpolating between continuous token relaxations
and hard tokens according to predictive entropy.

**Carnot-verifier implication:** EntRGi is the best concrete guidance mechanism
only after the verifier is known to be discriminative. It is not evidence that
the Carnot verifier already has value.

**Queued DiffusionGemma implication:** Use EntRGi's entropy-aware soft/hard
token interpolation as the DiffusionGemma implementation template after the
gate flips positive.

**Experiment mapping:** Keep EntRGi queued behind the .387 V-STaR/headroom gate
rather than launching it from the current deferred graft state.

## EDLM sequence-level diffusion energy comparator

**Method/source:** EDLM, arXiv:2410.21357
(https://arxiv.org/abs/2410.21357), adds a residual sequence-level energy model
to diffusion language modeling and uses parallel importance sampling.

**Carnot-verifier implication:** EDLM is the internal-energy comparator. Carnot
should measure whether the external executable verifier adds information beyond
diffusion-model energy.

**Queued DiffusionGemma implication:** For DiffusionGemma, compare external
Carnot energy with an internal sequence-level energy baseline, so guidance is
not merely relabeled EBM behavior.

**Experiment mapping:** Add EDLM-style internal-energy control after the
external verifier discrimination gate is satisfied.

## Flagged for .387

`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387` is the
strongest follow-on. Exp 4168 did not prove the verifier negative; it deferred
because the baseline was not yet faithful/stable. Therefore .387 should bank
paired accepted/rejected traces and a selector/headroom gate before activating
EntRGi-style DiffusionGemma energy guidance. EntRGi remains the strongest
guidance template, but it should stay queued unless the verifier discrimination
gate flips positive.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4170 - .387 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-v387-2026-06-13.md`.

**Filtered track:** verifier-as-reward, accepted/rejected trace selection, and
energy-guided generation for the `.387` handoff. This ingestion keeps
DiffusionGemma guidance queued because Exp 4168 recorded
`verifier_value_added=false` from a deferred, unfaithful/still-training
baseline rather than from a tested positive or negative guidance result.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful baseline and oracle-headroom
  gate before any verifier or diffusion-guidance claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget no-verifier adaptation
  control.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector and
  strongest `.387` next step.
- SEDD, arXiv:2310.16834 - mapped as the discrete score/energy scaffold for
  generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free guidance,
  arXiv:2207.12598 - mapped as the external-energy precedent and internal-score
  control.
- EntRGi, arXiv:2602.05000 - mapped as the queued DiffusionGemma reward-guidance
  template after a positive verifier-discrimination gate.
- EDLM, arXiv:2410.21357 - mapped as the internal sequence-energy comparator
  for any future guidance claim.

flagged_for_v387:
`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

Flagged for .387: `vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

**Bottom line for the .387 roadmap:** build the V-STaR-style rejected-trace
selector and headroom gate first. Keep EntRGi/DiffusionGemma guidance queued
unless the verifier discrimination gate flips positive.
"""

STUDYING_MARKER = (
    "## 2026-06-13 Exp 4170 - .387 verifier-moat guidance SOTA ingestion ingested"
)


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v387: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4170 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v387": flagged_for_v387,
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
                "carnot_verifier_implication, queued_diffusiongemma_implication, "
                "and experiment_mapping"
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

    flagged = artifact["flagged_for_v387"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v387 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "TRM nano-trm baseline and headroom gate",
        "TTA-TRM adaptation-control arm",
        "V-STaR accepted/rejected trace selector",
        "SEDD discrete diffusion score-energy formalism",
        "Classifier-guided diffusion external-energy precedent",
        "Classifier-free guidance control",
        "EntRGi entropy-aware reward guidance",
        "EDLM sequence-level diffusion energy comparator",
        "Carnot-verifier implication",
        "Queued DiffusionGemma implication",
        "Experiment mapping",
        "Flagged for .387",
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
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        flagged_for_v387=DEFAULT_FLAGGED_FOR_V387,
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
    """Write the default Exp 4170 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/"
        "sota-ingestion-verifier-moat-guidance-v387-2026-06-13.md",
        artifact_path=repo_root
        / "results/experiment_4170_sota_ingestion_verifier_moat_guidance.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
