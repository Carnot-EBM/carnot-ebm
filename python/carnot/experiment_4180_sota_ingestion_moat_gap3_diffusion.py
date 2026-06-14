"""Exp 4180 SOTA ingestion for moat, GAP-3, and diffusion planning.

Spec refs: REQ-REPORT-4180, SCENARIO-REPORT-4180.

This module writes a planning artifact, not a benchmark result. The purpose is
to close the discover->ingest->plan loop for the `.387 planning sweep`: each
paper is tied to a concrete Carnot experiment target and to the limitation that
keeps the mapping honest.
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
        "flagged_for_v388",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_moat_gap3_diffusion_mapped_v388"
DEFAULT_FLAGGED_FOR_V388 = "cem_gap3_stage2_compositional_arc_energy_v388"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify discipline)."
    ),
    "flagged_for_v388": (
        "Closes discover->ingest->plan: names the strongest method for the next planner."
    ),
}

VERIFIED_ARXIV_IDS = frozenset(
    {
        "2605.07395",
        "2504.01005",
        "2504.16828",
        "2510.20607",
        "2602.01849",
        "2512.11847",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{arxiv_id}" for arxiv_id in VERIFIED_ARXIV_IDS
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Unsolvability Ceiling headroom sanitization",
        "arxiv_id_or_url": "2605.07395",
        "url": "https://arxiv.org/abs/2605.07395",
        "carnot_stack_mapping": (
            "A1 headroom gate: keep the executable oracle, exact-format checks, "
            "and artifact-sanitized oracle@K measurement before treating any "
            "verifier delta as real headroom."
        ),
        "implication": (
            "A positive moat result must survive judge-bias, truncation, and "
            "format-mismatch controls; otherwise it may only be routed noise."
        ),
        "failure_mode": (
            "The paper audits routing headroom but does not provide a verifier "
            "or ARC energy mechanism; it is a measurement-discipline anchor."
        ),
        "experiment_mapping": (
            "Keep A1 as the mandatory precondition for A3 and GAP-3: no "
            "accuracy-cost claim without executable oracle headroom."
        ),
    },
    {
        "name": "When To Solve/Verify accuracy-cost moat",
        "arxiv_id_or_url": "2504.01005",
        "url": "https://arxiv.org/abs/2504.01005",
        "carnot_stack_mapping": (
            "A3 moat framing: report accuracy and verifier-call cost against "
            "self-consistency rather than reporting accuracy alone."
        ),
        "implication": (
            "Carnot's external energy must beat or sit on a Pareto frontier "
            "against scaled solution sampling, because GenRM-style verification "
            "can need much more compute before matching self-consistency."
        ),
        "failure_mode": (
            "It studies generative verification, not Carnot's cheap executable "
            "energy; using it as a direct baseline would overcharge the verifier."
        ),
        "experiment_mapping": (
            "In .388, keep the A3 table cost-normalized: vote@K, oracle@K, "
            "Carnot-rerank@K, verifier calls, and wall-clock budget."
        ),
    },
    {
        "name": "ThinkPRM process-verifier cost control",
        "arxiv_id_or_url": "2504.16828",
        "url": "https://arxiv.org/abs/2504.16828",
        "carnot_stack_mapping": (
            "A3 verifier comparator: use ThinkPRM as the high-quality process "
            "verifier reference when framing the accuracy-and-cost moat."
        ),
        "implication": (
            "The moat is conditional: a verifier can help under the right "
            "budget and process-label regime, but the claim must show value "
            "over both vote aggregation and an expensive PRM comparator."
        ),
        "failure_mode": (
            "ThinkPRM's long generative verification is a different cost class "
            "from Carnot energy; it can validate the target but not prove cheapness."
        ),
        "experiment_mapping": (
            "Use A3 to separate quality from cost: Carnot must report any "
            "accuracy lift alongside the verifier-cost discount against PRM judging."
        ),
    },
    {
        "name": "CEM compositional ARC energy",
        "arxiv_id_or_url": "2510.20607",
        "url": "https://arxiv.org/abs/2510.20607",
        "carnot_stack_mapping": (
            "GAP-3 Stage-2: learn local rule/content energies over ARC-style "
            "subproblems, compose them at inference, then sample with a PEM-like "
            "parallel minimization loop."
        ),
        "implication": (
            "This is the strongest .388 method because it targets the next "
            "unbuilt Carnot capability: a learned compositional energy for ARC "
            "transitions rather than only reranking finished candidates."
        ),
        "failure_mode": (
            "CEM is not an ARC-AGI-3 result and its benchmark energies are not "
            "Carnot's executable transition checks; transfer must be earned on "
            "the GAP-3 harness."
        ),
        "experiment_mapping": (
            "Flag .388 for a Stage-2 compositional ARC energy prototype: train "
            "factor energies on small transformations, compose on held-out tasks, "
            "and compare PEM sampling to the current GAP-3 candidates."
        ),
    },
    {
        "name": "Self-Rewarding SMC DiffusionGemma guidance",
        "arxiv_id_or_url": "2602.01849",
        "url": "https://arxiv.org/abs/2602.01849",
        "carnot_stack_mapping": (
            "DiffusionGemma guidance: use particle weighting/resampling over "
            "masked-diffusion trajectories as the training-free guidance template."
        ),
        "implication": (
            "A guidance experiment can convert parallel denoising capacity into "
            "better candidates without training a new reward model, which fits "
            "the gated DiffusionGemma plan."
        ),
        "failure_mode": (
            "The reward is trajectory confidence, not an external executable "
            "Carnot verifier; it may improve fluency or confidence without "
            "improving task correctness."
        ),
        "experiment_mapping": (
            "Keep it queued behind the verifier gate: after A3 shows positive "
            "energy discrimination, test SMC-style particle guidance on "
            "DiffusionGemma with exact downstream validation."
        ),
    },
    {
        "name": "TRM ARC headroom-vote decomposition",
        "arxiv_id_or_url": "2512.11847",
        "url": "https://arxiv.org/abs/2512.11847",
        "carnot_stack_mapping": (
            "TRM headroom/vote decomposition: separate single-pass accuracy, "
            "1000-sample vote gains, identity conditioning, and candidate "
            "coverage before crediting a verifier."
        ),
        "implication": (
            "The TRM stack needs an oracle@K versus vote@1 decomposition; if "
            "vote is carrying the result, Carnot should not claim verifier value."
        ),
        "failure_mode": (
            "The ablation is a warning about TRM substrate artifacts, not a "
            "recipe for Carnot energy or DiffusionGemma guidance."
        ),
        "experiment_mapping": (
            "For .388, keep TRM as a controlled generator: report identity-ID "
            "ablations and vote/headroom decomposition before any rerank result."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-14: moat, GAP-3, and diffusion map for .388

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_moat_gap3_diffusion_mapped_v388`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Unsolvability Ceiling headroom sanitization`, arxiv_id_or_url: `2605.07395`, url: `https://arxiv.org/abs/2605.07395`}
  - {name: `When To Solve/Verify accuracy-cost moat`, arxiv_id_or_url: `2504.01005`, url: `https://arxiv.org/abs/2504.01005`}
  - {name: `ThinkPRM process-verifier cost control`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `CEM compositional ARC energy`, arxiv_id_or_url: `2510.20607`, url: `https://arxiv.org/abs/2510.20607`}
  - {name: `Self-Rewarding SMC DiffusionGemma guidance`, arxiv_id_or_url: `2602.01849`, url: `https://arxiv.org/abs/2602.01849`}
  - {name: `TRM ARC headroom-vote decomposition`, arxiv_id_or_url: `2512.11847`, url: `https://arxiv.org/abs/2512.11847`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v388: `cem_gap3_stage2_compositional_arc_energy_v388`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-references.md` `.387 planning sweep` and the
`research-studying.md` / `research-references.md` verifier-as-reward,
headroom, and energy-guided diffusion entries.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier as reward headroom compute optimal verification process reward model" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "compositional energy minimization ARC masked diffusion self rewarding SMC" --limit 8`

The cluster helper emitted the broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for both focused queries during this run, so
no fresh S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch
verified the requested mapped paper set: arXiv:2605.07395, arXiv:2504.01005,
arXiv:2504.16828, arXiv:2510.20607, arXiv:2602.01849, and arXiv:2512.11847.

## SOTA -> experiment mapping

## Unsolvability Ceiling headroom sanitization

**Method/source:** Unsolvability Ceiling, arXiv:2605.07395
(https://arxiv.org/abs/2605.07395), audits multi-LLM routing headroom and
shows that judge bias, truncation, and output-format mismatch can inflate the
apparent oracle gap.

**Carnot stack mapping:** This maps to the A1 headroom-gate sanitization
already applied: use executable or exact objective checks, retain oracle@K,
and reject unsanitized judge-only headroom.

**Implication:** A positive verifier result is only decision-grade if it
survives objective oracle checks and cost-sensitive routing controls.

**Failure mode:** This paper does not provide a verifier, ARC energy, or
DiffusionGemma mechanism. It only tells us how the measurement can lie.

**Experiment mapping:** Keep A1 as a mandatory precondition for A3 and GAP-3:
no accuracy-cost moat claim without executable headroom.

## When To Solve/Verify accuracy-cost moat

**Method/source:** When To Solve, When To Verify, arXiv:2504.01005
(https://arxiv.org/abs/2504.01005), compares self-consistency with generative
verification under fixed inference budgets.

**Carnot stack mapping:** This maps to A3's accuracy-and-cost moat framing:
report vote@K, oracle@K, Carnot rerank@K, verifier calls, and wall-clock cost.

**Implication:** Carnot must prove that a cheap executable energy beats or
sits on the Pareto frontier against simply sampling more solutions.

**Failure mode:** The paper studies GenRM-style generative verification, not
Carnot's executable energy. It sets a cost bar but is not a direct substrate.

**Experiment mapping:** Keep the A3 table cost-normalized and do not report an
accuracy-only moat result.

## ThinkPRM process-verifier cost control

**Method/source:** ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), uses generative process verification with
small synthetic supervision to judge reasoning steps.

**Carnot stack mapping:** This also maps to A3: it is the high-quality PRM
comparator for an accuracy-and-cost moat, not the cheap-energy method itself.

**Implication:** Verifier advantage is real but budget-conditional; Carnot must
show both quality and cost separation from an expensive process verifier.

**Failure mode:** ThinkPRM's long generative judging can be too expensive for
Carnot's claimed moat, so it validates the target class without proving our
efficiency claim.

**Experiment mapping:** Use ThinkPRM as the expensive quality comparator while
Carnot energy must carry the cheap verifier arm.

## CEM compositional ARC energy

**Method/source:** Generalizable Reasoning through Compositional Energy
Minimization, arXiv:2510.20607 (https://arxiv.org/abs/2510.20607), learns
subproblem energy landscapes and composes them at test time.

**Carnot stack mapping:** This is the GAP-3 Stage-2 compositional ARC energy
map: factor rule/content energies, compose them on held-out tasks, and sample
with a PEM-like parallel minimization loop.

**Implication:** This is the strongest .388 method because it targets the next
unbuilt Carnot capability: a learned transition energy, not another post-hoc
reranker.

**Failure mode:** CEM is not an ARC-AGI-3 result and not a Carnot executable
transition verifier. Transfer must be measured on the GAP-3 harness.

**Experiment mapping:** Flag .388 for `cem_gap3_stage2_compositional_arc_energy_v388`:
train local transformation energies, compose them on held-out ARC-style tasks,
and compare PEM sampling against the current GAP-3 candidates.

## Self-Rewarding SMC DiffusionGemma guidance

**Method/source:** Self-Rewarding SMC for Masked Diffusion Language Models,
arXiv:2602.01849 (https://arxiv.org/abs/2602.01849), weights and resamples
parallel masked-diffusion particles using trajectory confidence.

**Carnot stack mapping:** This maps to the queued DiffusionGemma guidance plan:
use particle weighting and resampling as the training-free guidance template.

**Implication:** A future DiffusionGemma run can convert parallel denoising
capacity into better candidates without training a separate reward model.

**Failure mode:** The reward is model confidence, not external executable
correctness. It can improve confident sampling without improving task validity.

**Experiment mapping:** Keep SMC-style guidance queued behind A3: use it for
DiffusionGemma guidance once the energy gate is positive.

## TRM ARC headroom-vote decomposition

**Method/source:** Tiny Recursive Models on ARC-AGI-1, arXiv:2512.11847
(https://arxiv.org/abs/2512.11847), decomposes TRM performance into voting,
identity conditioning, and shallow recursion effects.

**Carnot stack mapping:** This maps to the TRM headroom/vote decomposition:
separate single-pass, vote@1000, oracle@K, and identity-ID ablation before
crediting a verifier.

**Implication:** TRM can remain the candidate generator, but only with a clean
headroom/vote receipt that shows where selection value could exist.

**Failure mode:** The ablation is a warning about substrate artifacts, not an
energy or guidance mechanism.

**Experiment mapping:** Keep TRM as a controlled generator and require identity
conditioning plus vote/headroom ablations before any rerank claim.

## Flagged for .388

`cem_gap3_stage2_compositional_arc_energy_v388` is the strongest follow-on.
A1 headroom sanitation and A3 accuracy-cost framing are necessary gates, and
Self-Rewarding SMC is a useful DiffusionGemma template after the verifier gate
turns positive. The method that most directly advances Carnot's stack is CEM:
it gives GAP-3 Stage-2 a concrete learned compositional energy experiment.
"""

STUDYING_SECTION = """## 2026-06-14 Exp 4180 - .387 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-moat-gap3-diffusion-v388-2026-06-14.md`.

**Filtered track:** verifier-as-reward, sanitized headroom, accuracy-and-cost
moat framing, GAP-3 learned ARC energy, TRM vote/headroom decomposition, and
DiffusionGemma guidance for the `.388` handoff.

**Seed and fresh-pass candidates marked ingested:**
- Unsolvability Ceiling, arXiv:2605.07395 - mapped to the A1 headroom-gate
  sanitization already applied; it is a measurement guard, not a verifier.
- When To Solve/Verify, arXiv:2504.01005 - mapped to A3 accuracy-and-cost
  reporting against self-consistency.
- ThinkPRM, arXiv:2504.16828 - mapped to A3 as the high-quality but expensive
  process-verifier comparator.
- Generalizable Reasoning through Compositional Energy Minimization,
  arXiv:2510.20607 - mapped to GAP-3 Stage-2 compositional ARC energy and
  flagged as the strongest `.388` follow-on.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped to the queued DiffusionGemma
  particle-guidance template after a positive energy gate.
- TRM ARC-AGI-1 ablation, arXiv:2512.11847 - mapped to the TRM headroom/vote
  decomposition and identity-conditioning control.

flagged_for_v388:
`cem_gap3_stage2_compositional_arc_energy_v388`.

Flagged for .388: `cem_gap3_stage2_compositional_arc_energy_v388`.

**Bottom line for the .388 roadmap:** run the CEM-style GAP-3 Stage-2
compositional ARC energy prototype first. Keep A1/A3 as mandatory gates and use
Self-Rewarding SMC only for DiffusionGemma guidance once the energy gate is positive.
"""

STUDYING_MARKER = "## 2026-06-14 Exp 4180 - .387 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v388: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4180 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v388": flagged_for_v388,
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
                "carnot_stack_mapping, implication, failure_mode, and experiment_mapping"
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

    flagged = artifact["flagged_for_v388"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v388 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "Unsolvability Ceiling headroom sanitization",
        "When To Solve/Verify accuracy-cost moat",
        "ThinkPRM process-verifier cost control",
        "CEM compositional ARC energy",
        "Self-Rewarding SMC DiffusionGemma guidance",
        "TRM ARC headroom-vote decomposition",
        "Carnot stack mapping",
        "Implication",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .388",
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
        flagged_for_v388=DEFAULT_FLAGGED_FOR_V388,
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
    """Write the default Exp 4180 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-moat-gap3-diffusion-v388-2026-06-14.md",
        artifact_path=repo_root
        / "results/experiment_4180_sota_ingestion_moat_gap3_diffusion.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
