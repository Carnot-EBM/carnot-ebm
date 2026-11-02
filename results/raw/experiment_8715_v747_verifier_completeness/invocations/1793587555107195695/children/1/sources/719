"""Exp 4276 SOTA ingestion for the .395 fork outcomes feeding .396.

Spec refs: REQ-REPORT-4276, SCENARIO-REPORT-4276.

This module writes a planning artifact, not a benchmark result. It closes the
`.395` cross-family ARC transfer outcome into a concrete SOTA-to-experiment
mapping: the selector generalized to held-out families, so .396 can spend the
opened scale-up budget on a bounded DiffusionGemma guidance full run while also
stress-testing the generalization claim with a second procedural ARC family
source and guarded online adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v396",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "track",
        "source_read",
        "v395_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v396_mapped"
DEFAULT_FLAGGED_FOR_V396 = (
    "rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396"
)
COLLAPSED_FLAGGED_FOR_V396 = "generalizing_verifier_meta_feature_repair_v396"
DEFAULT_RANDOM_SEED = 4276

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .396 experiment mapping."
    ),
    "flagged_for_v396": (
        "Closes discover->ingest->plan: names the strongest method for the .396 "
        "planner, conditioned on whether cross-family generalized or collapsed."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2601.18217": "https://arxiv.org/abs/2601.18217",
    "2511.00162": "https://arxiv.org/abs/2511.00162",
    "2509.25604": "https://arxiv.org/abs/2509.25604",
    "2510.07841": "https://arxiv.org/abs/2510.07841",
    "2603.25111": "https://arxiv.org/abs/2603.25111",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Paying Less Generalization Tax cross-domain RL recipe",
        "arxiv_id_or_url": "2601.18217",
        "url": "https://arxiv.org/abs/2601.18217",
        "track": "cross-family verifier generalization",
        "source_read": (
            "The paper studies RL agents deployed on unknown domains and finds "
            "that state-information richness, planning complexity, and "
            "step-by-step thinking matter more for transfer than superficial "
            "domain realism or text similarity."
        ),
        "v395_outcome_conditioning": (
            "Exp 4271 reports cross_family_generalizes, so .396 should not "
            "treat the selector as a within-pool artifact. It should widen the "
            "stress test with richer family metadata and distractor-style "
            "invariance checks."
        ),
        "carnot_stack_mapping": (
            "Add family-rich but label-irrelevant ARC meta-features and "
            "controlled distractor channels to the held-out-family split, then "
            "verify that the Set-Encoder lift remains positive when these "
            "features are randomized or removed."
        ),
        "failure_mode": (
            "Environment randomization can create robustness theater if the "
            "randomized feature is not causally tied to the held-out rule. Keep "
            "original-task, ARC-TGI, and ARC-GEN reads separate."
        ),
        "experiment_mapping": (
            ".396: strengthen the cross-family headline with randomized "
            "family-rich stress splits and report selector lift separately on "
            "original ARC, ARC-TGI, and ARC-GEN families."
        ),
    },
    {
        "name": "ARC-GEN mimetic procedural ARC generator",
        "arxiv_id_or_url": "2511.00162",
        "url": "https://arxiv.org/abs/2511.00162",
        "track": "procedural ARC family-generator follow-up",
        "source_read": (
            "ARC-GEN is an open-source procedural generator intended to extend "
            "the original ARC-AGI training set faithfully, covering all 400 "
            "ARC-AGI-1 tasks with a mimetic generator design."
        ),
        "v395_outcome_conditioning": (
            "Because Exp 4271 generalized on recovered original-task families, "
            ".396 needs a second generator substrate to check that the result "
            "is not specific to the ARC-TGI or recovered-manifest partition."
        ),
        "carnot_stack_mapping": (
            "Build a second family-disjoint candidate pool from ARC-GEN tasks, "
            "materialize generator IDs and target hashes, and rerun the "
            "Set-Encoder versus vote gate with the same bootstrap protocol."
        ),
        "failure_mode": (
            "Mimetic generators can clone original task quirks. The split must "
            "report generator-family, original-task, and mixed-source metrics "
            "rather than pooling them into one transfer number."
        ),
        "experiment_mapping": (
            ".396: run an ARC-GEN transfer stress gate as an independent "
            "procedural-family replication of the Exp 4271 generalization win."
        ),
    },
    {
        "name": "RFG reward-free guidance for diffusion LLM reasoning",
        "arxiv_id_or_url": "2509.25604",
        "url": "https://arxiv.org/abs/2509.25604",
        "track": "discrete-diffusion verifier and guidance full run",
        "source_read": (
            "RFG guides diffusion LLM reasoning without an explicit process "
            "reward by using log-likelihood ratios between enhanced and "
            "reference diffusion models, improving math and code benchmarks."
        ),
        "v395_outcome_conditioning": (
            "The cross-family selector generalized, and the .395 loader path "
            "moves DiffusionGemma from deferred smoke toward a full run. RFG is "
            "the safest full-run anchor when explicit ARC process rewards are "
            "still sparse or noisy."
        ),
        "carnot_stack_mapping": (
            "Pair loader-fixed DiffusionGemma with a reference unguided pass, "
            "apply reward-free likelihood-ratio guidance at denoising time, and "
            "use the generalized selector only as the final exact-grid arbiter."
        ),
        "failure_mode": (
            "Reward-free guidance can amplify a post-trained model's biases and "
            "may not optimize ARC exact match. It needs unguided, selector-only, "
            "DPRM-style, and EntRGi-style controls."
        ),
        "experiment_mapping": (
            ".396: run a bounded DiffusionGemma RFG full-run arm against "
            "unguided diffusion, DPRM/EntRGi-inspired guidance controls, and "
            "the existing selector-only baseline."
        ),
    },
    {
        "name": "Self-Improving LLM Agents at Test-Time",
        "arxiv_id_or_url": "2510.07841",
        "url": "https://arxiv.org/abs/2510.07841",
        "track": "online adaptation and test-time self-improvement",
        "source_read": (
            "The method identifies uncertain cases, generates similar examples, "
            "and uses them for test-time fine-tuning or teacher-distilled "
            "adaptation, reporting average gains with far fewer samples."
        ),
        "v395_outcome_conditioning": (
            "Since Exp 4271 already shows a static selector transfer signal, "
            ".396 can test whether uncertainty-triggered adaptation improves "
            "held-out families without turning every task into per-family memory."
        ),
        "carnot_stack_mapping": (
            "Trigger adaptation only on low-margin held-out-family tasks, create "
            "synthetic same-rule variants from the family generator, fine-tune "
            "only a tiny selector head, and compare to a frozen-selector control."
        ),
        "failure_mode": (
            "Test-time adaptation can leak target structure or overfit a single "
            "family. It must keep target outputs hidden, use adaptation budgets, "
            "and report when frozen raw trajectories outperform distilled updates."
        ),
        "experiment_mapping": (
            ".396: add a low-margin TT-SI selector-head adaptation arm on "
            "held-out ARC-GEN families with frozen-selector and random-update "
            "controls."
        ),
    },
    {
        "name": "SEVerA verified synthesis for self-evolving agents",
        "arxiv_id_or_url": "2603.25111",
        "url": "https://arxiv.org/abs/2603.25111",
        "track": "guarded online selector self-improvement",
        "source_read": (
            "SEVerA wraps self-evolving agent programs with formal output "
            "contracts and verified fallbacks, then learns soft objectives while "
            "preserving the hard constraints."
        ),
        "v395_outcome_conditioning": (
            "A generalized selector makes online improvement tempting, but .396 "
            "needs hard guardrails before any self-modifying selector or "
            "diffusion refiner can affect ARC outputs."
        ),
        "carnot_stack_mapping": (
            "Wrap any online selector/refiner update in a contract that checks "
            "grid shape, palette, immutable train examples, and exact target-hash "
            "provenance, with fallback to the frozen generalized selector."
        ),
        "failure_mode": (
            "Formal contracts can enforce syntax and provenance but not semantic "
            "ARC correctness. They reduce unsafe updates; they do not replace "
            "held-out exact-match gates."
        ),
        "experiment_mapping": (
            ".396: guard TT-SI and DiffusionGemma refinement arms with verified "
            "fallback contracts before counting any online-adaptation lift."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-16: .395 forks map for .396

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_v396_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Paying Less Generalization Tax cross-domain RL recipe`, arxiv_id_or_url: `2601.18217`, url: `https://arxiv.org/abs/2601.18217`}
  - {name: `ARC-GEN mimetic procedural ARC generator`, arxiv_id_or_url: `2511.00162`, url: `https://arxiv.org/abs/2511.00162`}
  - {name: `RFG reward-free guidance for diffusion LLM reasoning`, arxiv_id_or_url: `2509.25604`, url: `https://arxiv.org/abs/2509.25604`}
  - {name: `Self-Improving LLM Agents at Test-Time`, arxiv_id_or_url: `2510.07841`, url: `https://arxiv.org/abs/2510.07841`}
  - {name: `SEVerA verified synthesis for self-evolving agents`, arxiv_id_or_url: `2603.25111`, url: `https://arxiv.org/abs/2603.25111`}
  - principle: Each method MUST carry a real arXiv ID/URL (no citation = fabrication per adversarial_verify discipline) + a one-line .396 experiment mapping.
- flagged_for_v396: `rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`
  - principle: Closes discover->ingest->plan: names the strongest method for the .396 planner, conditioned on whether cross-family generalized or collapsed.
- random_seed: `4276`
  - principle: Determinism placeholder for the discovery query set (recorded for reproducibility of the sweep).

## Fresh-pass provenance

Read `CLAUDE.md` SOTA-Ingestion Cycle Discipline, `research-studying.md`,
`research-references.md`, `results/experiment_4265_sota_ingestion_v395.json`,
`results/experiment_4271_arc_cross_family_transfer_existing_pool.json`,
`results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 -c "import importlib; importlib.import_module('scripts.sweep_clusters'); importlib.import_module('scripts.sweep_semscholar')"`
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_clusters.py all --max-results 8`
- `python3 scripts/sweep_semscholar.py "cross task verifier generalization learned selector out of domain test time adaptation" --limit 8`
- `python3 scripts/sweep_semscholar.py "diffusion language model process reward verifier guidance discrete diffusion DPRM" --limit 8`

The sweep helpers imported successfully and `sweep_clusters.py` emitted the
broadened arXiv API URLs for the reliable channel. The local Semantic Scholar
helper was reachable as code but degraded at fetch time due to TLS certificate
verification failure, so no Semantic-Scholar-only promotion is claimed.
WebSearch/WebFetch was reachable and verified arXiv:2601.18217,
arXiv:2511.00162, arXiv:2509.25604, arXiv:2510.07841, arXiv:2603.25111,
plus prior-covered context arXiv:2604.24357, arXiv:2602.05000, and
arXiv:2603.05099. The banned `/deep-research` channel was not invoked.

## Prior-covered methods not re-ingested

The .394/.395 sweeps already covered ARC-TGI (arXiv:2603.05099), Reliability
Gap (arXiv:2606.03305), DPRM (arXiv:2604.24357), entropy-guided diffusion RL
(arXiv:2603.12554), L-VARC (arXiv:2606.12847), TrajAD (arXiv:2602.06443),
RL^V / Putting the Value Back in RL (arXiv:2505.04842), EntRGi
(arXiv:2602.05000), and Self-Trained Verification (arXiv:2605.30290). They
remain context for .396, but they are not counted as fresh `methods_mapped`
rows here.

## .395 cross-family outcome read

Exp 4271: `cross_family_generalizes`, `cross_family_win_holds=true`,
`cross_family_delta=0.4038461538`, `cross_family_ci95=[0.25, 0.5576923077]`,
`held_out_family_n=52`, `held_out_task_n=52`, and `verifier_is_oracle=false`.
The hardened Set-Encoder selector survived the load-bearing OOD gate, so .396
can treat the selector as a general transfer signal rather than within-pool
memorization.

Exp 4272 was correctly blocked because Exp 4270 found
`family_split_feasible=true` for the existing pool; the fresh ARC-TGI fallback
was not needed for .395. This means .396 should use fresh procedural generators
as a stronger stress test, not as a repair for a failed .395 gate.

## SOTA -> experiment mapping

## Paying Less Generalization Tax: stress the transfer claim

**Method/source:** Paying Less Generalization Tax, arXiv:2601.18217
(https://arxiv.org/abs/2601.18217), studies cross-domain RL transfer and
identifies state-information richness, planning complexity, and step-by-step
thinking as stronger transfer drivers than surface domain similarity.

**Carnot stack mapping:** Add family-rich but label-irrelevant ARC meta-features
and randomized distractor channels to the held-out-family split, then check
whether the generalized Set-Encoder lift survives removal or randomization of
those channels.

**.395 conditioning:** Because Exp 4271 generalized, .396 should deepen the
generalization headline rather than repair a collapse.

**Failure mode:** Randomization can become a cosmetic robustness test if the
added feature does not touch the causal family rule. Keep original ARC,
ARC-TGI, and ARC-GEN metrics separate.

**Experiment mapping:** .396 adds a richer cross-family stress split and reports
selector lift separately by source family.

## ARC-GEN: independent procedural-family replication

**Method/source:** ARC-GEN, arXiv:2511.00162
(https://arxiv.org/abs/2511.00162), is a mimetic procedural benchmark generator
covering all 400 ARC-AGI-1 tasks.

**Carnot stack mapping:** Build a second family-disjoint candidate pool from
ARC-GEN, materialize generator IDs and target hashes, and rerun the same
Set-Encoder versus vote bootstrap gate.

**.395 conditioning:** Exp 4271 generalized on recovered original-task family
IDs. ARC-GEN checks whether the transfer survives a different procedural
family substrate rather than only the recovered-manifest split.

**Failure mode:** Mimetic generation can clone original task quirks. Report
original-task and generated-family metrics separately.

**Experiment mapping:** .396 runs ARC-GEN as the independent transfer stress
gate after the .395 generalization win.

## RFG: bounded DiffusionGemma full run

**Method/source:** RFG, arXiv:2509.25604
(https://arxiv.org/abs/2509.25604), guides diffusion LLM reasoning with
log-likelihood ratios between enhanced and reference diffusion models instead
of explicit process-reward labels.

**Carnot stack mapping:** Pair loader-fixed DiffusionGemma with a reference
unguided pass, apply reward-free guidance at denoising time, and use the
generalized selector as the final exact-grid arbiter.

**.395 conditioning:** The cross-family selector generalized, so .396 can spend
the diffusion scale-up budget. RFG is the lowest-label-debt full-run method
when ARC process rewards remain sparse.

**Failure mode:** RFG can amplify the enhanced model's biases and may not align
with ARC exact match. Compare to unguided diffusion, selector-only, DPRM-style,
and EntRGi-style controls.

**Experiment mapping:** .396 runs a bounded DiffusionGemma RFG arm with exact
grid validation and cost-normalized controls.

## Test-Time Self-Improvement: low-margin selector adaptation

**Method/source:** Self-Improving LLM Agents at Test-Time, arXiv:2510.07841
(https://arxiv.org/abs/2510.07841), identifies uncertain cases, generates
similar examples, and adapts at test time from those examples.

**Carnot stack mapping:** Trigger adaptation only on low-margin held-out-family
tasks, create synthetic same-rule variants from the procedural generator,
fine-tune a tiny selector head, and compare to frozen-selector and random-update
controls.

**.395 conditioning:** A static selector already generalized, so adaptation is
an optional lift test, not a rescue path.

**Failure mode:** Adaptation can leak target structure or overfit one family.
Keep target outputs hidden and cap the adaptation budget.

**Experiment mapping:** .396 adds a low-margin TT-SI selector-head adaptation
arm on held-out ARC-GEN families.

## SEVerA: verified fallback for self-improving branches

**Method/source:** SEVerA, arXiv:2603.25111
(https://arxiv.org/abs/2603.25111), combines formal output contracts, verified
fallbacks, and learning over soft objectives for self-evolving agents.

**Carnot stack mapping:** Wrap online selector/refiner updates in contracts that
check grid shape, palette, immutable train examples, and provenance hashes, then
fallback to the frozen generalized selector on contract failure.

**.395 conditioning:** The selector generalized, but any .396 online
self-improvement branch needs guardrails before it can affect reported ARC
outputs.

**Failure mode:** Contracts enforce syntax and provenance, not semantic ARC
correctness. They are a safety guard, not a replacement for exact-match gates.

**Experiment mapping:** .396 guards TT-SI and DiffusionGemma refinement arms
with verified fallback contracts before counting online-adaptation lift.

## Flagged for .396

`rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396` is the strongest
next method because the condition is the positive one: Exp 4271 reports
`cross_family_generalizes` with `cross_family_win_holds=true`,
`cross_family_delta=0.4038461538`, and CI95 `[0.25, 0.5576923077]`. That opens
the scale-up gate. The .396 planner should run a bounded RFG-style
DiffusionGemma full-run arm using the generalized selector as the exact-grid
arbiter, while ARC-GEN supplies the independent procedural-family stress test.
If the cross-family result had collapsed, the correct flag would have been
`generalizing_verifier_meta_feature_repair_v396`; it did not collapse.

random_seed=4276
"""

STUDYING_SECTION = """## 2026-06-16 Exp 4276 - .395 fork SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-v396-2026-06-16.md`.

**Filtered track:** .395 ARC cross-family transfer after the hardened
oracle-distinct Set-Encoder selector generalized on held-out families
(`cross_family_delta=0.4038461538`, CI95 `[0.25, 0.5576923077]`) while the
fresh ARC-TGI fallback was correctly gate-blocked because the existing family
split was feasible.

**Fresh-pass candidates marked ingested:**
- Paying Less Generalization Tax, arXiv:2601.18217 - mapped to a stronger
  cross-family stress split with richer randomized family metadata.
- ARC-GEN, arXiv:2511.00162 - mapped to an independent procedural-family
  replication of the Exp 4271 transfer win.
- RFG, arXiv:2509.25604 - mapped as the queued DiffusionGemma full-run method
  now that cross-family selector generalization opened the scale-up gate.
- Self-Improving LLM Agents at Test-Time, arXiv:2510.07841 - mapped to
  low-margin selector-head adaptation on held-out generated families.
- SEVerA, arXiv:2603.25111 - mapped to verified fallback contracts for any
  self-improving selector or diffusion-refiner branch.

Already-covered context not re-ingested as fresh method rows: ARC-TGI,
Reliability Gap, DPRM, entropy-guided diffusion RL, L-VARC, TrajAD, RL^V,
EntRGi, and Self-Trained Verification.

.395 status mapped honestly: Exp 4271 `cross_family_generalizes` with
`cross_family_win_holds=true`, `cross_family_delta=0.4038461538`,
`cross_family_ci95=[0.25, 0.5576923077]`, `held_out_family_n=52`, and
`verifier_is_oracle=false`; Exp 4272 was blocked because the existing-pool
family split was feasible.

flagged_for_v396:
`rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`.

Flagged for .396: `rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`.

**Bottom line for the .396 roadmap:** cross-family did GENERALIZE, so run the
bounded RFG-style DiffusionGemma full-run arm with exact-grid selector
arbitration, and use ARC-GEN to independently stress the transfer claim while
keeping a stronger generalization stress test.
"""


def select_flagged_for_v396(cross_family_generalized: bool) -> str:
    """Choose the .396 flag from the cross-family transfer outcome."""

    if cross_family_generalized:
        return DEFAULT_FLAGGED_FOR_V396
    return COLLAPSED_FLAGGED_FOR_V396


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v396: str = DEFAULT_FLAGGED_FOR_V396,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4276 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v396": flagged_for_v396,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4276 artifact before it can be written to disk."""

    fields = set(artifact)
    missing = REQUIRED_ARTIFACT_FIELDS - fields
    extra = fields - REQUIRED_ARTIFACT_FIELDS
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-4276")

    random_seed = artifact["random_seed"]
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

    seen_sources: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must be a dict with exactly the required fields")

        for key, value in method.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method field {key!r} must be a non-empty string")

        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method source {source!r} is not a verified source")
        if method["url"] != VERIFIED_SOURCE_URLS[source]:
            raise ValueError(f"method url for {source!r} must match the verified url")
        if source in seen_sources:
            raise ValueError(f"duplicate source in methods_mapped: {source}")
        seen_sources.add(source)

    flagged = artifact["flagged_for_v396"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v396 must be non-empty")
    flagged_lower = flagged.lower()
    if "rfg_diffusiongemma" not in flagged_lower and "generalizing_verifier" not in flagged_lower:
        raise ValueError("flagged_for_v396 must be conditioned on the cross-family outcome")


def validate_markdown_note(note: str) -> None:
    """Check that the human-readable note keeps citations and outcome context."""

    required_phrases = [
        "Flagged for .396",
        "Prior-covered methods not re-ingested",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "TLS certificate",
        "cross_family_generalizes",
        "cross_family_win_holds=true",
        "cross_family_delta=0.4038461538",
        "cross_family_ci95=[0.25, 0.5576923077]",
        "held_out_family_n=52",
        "verifier_is_oracle=false",
        DEFAULT_FLAGGED_FOR_V396,
        f"random_seed={DEFAULT_RANDOM_SEED}",
    ]
    for phrase in required_phrases:
        if phrase not in note:
            raise ValueError(f"markdown note missing required phrase: {phrase}")

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in note
    )
    if missing_sources:
        raise ValueError(f"markdown note missing verified source citations: {missing_sources}")


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-16 Exp 4276"
    next_marker = "\n## "
    section = STUDYING_SECTION.rstrip() + "\n"

    if marker in existing:
        start = existing.index(marker)
        next_start = existing.find(next_marker, start + 1)
        if next_start == -1:
            return existing[:start] + section
        return existing[:start] + section + existing[next_start:]

    if existing.startswith("## "):
        return section + "\n" + existing

    first_section = existing.find(next_marker)
    if first_section == -1:
        return existing.rstrip() + "\n\n" + section
    return existing[: first_section + 1] + section + "\n" + existing[first_section + 1 :]


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent research-studying entry."""

    artifact = build_artifact()
    validate_artifact(artifact)
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN, encoding="utf-8")

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    existing_studying = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4276_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root / "docs/research-notes/sota-ingestion-v396-2026-06-16.md",
        artifact_path=repo_root / "results/experiment_4276_sota_ingestion_v396.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
