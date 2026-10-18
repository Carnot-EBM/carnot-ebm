"""Schema helpers for the Exp 4130 resumable-training SOTA ingestion.

Spec refs: REQ-REPORT-4130, SCENARIO-REPORT-4130.

This module records a planning artifact, not a training result. The trap in a
bounded long-running training loop is that a checkpoint can reload while the
learning-rate schedule, optimizer step, gradient-accumulation accounting, or
optimizer state silently restarts. That makes a run look long-horizon when it
is actually a sequence of short fresh starts. The validators below keep every
mapped source cite-backed and force the next planner to treat schedule
continuity as a measured precondition.
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
        "flagged_for_v383",
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
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_resumable_training_mapped"
DEFAULT_FLAGGED_FOR_V383 = "lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method/source MUST carry a real arXiv ID or canonical doc URL; "
        "an ingestion note without verifiable citations is treated as fabrication."
    ),
    "flagged_for_v383": (
        "Closes the discover->ingest->plan loop: names the strongest method "
        "for the next planner."
    ),
}

VERIFIED_ARXIV_IDS = frozenset({"2510.04871", "2511.02886"})
VERIFIED_CANONICAL_DOC_URLS = frozenset(
    {
        "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
        "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
    }
)
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    {
        "arXiv:2510.04871",
        "arXiv:2511.02886",
        "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
        "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
    }
)

DEFAULT_METHODS_MAPPED = [
    {
        "name": "PyTorch Lightning full-state checkpoint resume",
        "arxiv_id_or_url": "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
        "url": "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
        "implementation_over_stack": (
            "Resume nano-trm bounded passes with Trainer.fit(..., ckpt_path=...) "
            "so global_step, optimizer state, and LR scheduler state continue."
        ),
        "failure_mode": (
            "Loading weights only or restarting the trainer silently rewinds the "
            "warmup/cosine schedule and fabricates long-horizon progress."
        ),
    },
    {
        "name": "PyTorch optimizer-state checkpoint contract",
        "arxiv_id_or_url": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "url": "https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html",
        "implementation_over_stack": (
            "If a run bypasses Lightning, persist model state, optimizer state, "
            "epoch or step, LR scheduler state, data cursor, and RNG receipt."
        ),
        "failure_mode": (
            "A model-only checkpoint loses momentum buffers and parameter-group "
            "learning rates, causing an unreported resume discontinuity."
        ),
    },
    {
        "name": "Lightning gradient-accumulation schedule",
        "arxiv_id_or_url": "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
        "url": "https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html",
        "implementation_over_stack": (
            "Use accumulate_grad_batches or GradientAccumulationScheduler to keep "
            "effective batch size explicit across microbatches and bounded passes."
        ),
        "failure_mode": (
            "Counting microbatches as optimizer steps overstates the training "
            "horizon and moves the LR schedule at the wrong rate."
        ),
    },
    {
        "name": "TRM long-horizon baseline gate",
        "arxiv_id_or_url": "2510.04871",
        "url": "https://arxiv.org/abs/2510.04871",
        "implementation_over_stack": (
            "Accumulate resumed nano-trm Sudoku evidence by global optimizer step "
            "and held-out exact accuracy before any verifier-lift claim."
        ),
        "failure_mode": (
            "A checkpoint can reload correctly while still being an undertrained "
            "partial baseline, especially if LR state was reset."
        ),
    },
    {
        "name": "TTA-TRM bounded full-fine-tune control",
        "arxiv_id_or_url": "2511.02886",
        "url": "https://arxiv.org/abs/2511.02886",
        "implementation_over_stack": (
            "Keep a no-verifier full-fine-tune control with the same accumulated "
            "optimizer-step budget and the same resumed LR schedule receipts."
        ),
        "failure_mode": (
            "Full fine-tuning can win through adaptation compute alone, and a "
            "per-pass LR reset confounds the verifier comparison."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-13: resumable training with LR-schedule continuity

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_resumable_training_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `PyTorch Lightning full-state checkpoint resume`, arxiv_id_or_url: `https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html`, url: `https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html`}
  - {name: `PyTorch optimizer-state checkpoint contract`, arxiv_id_or_url: `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`, url: `https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html`}
  - {name: `Lightning gradient-accumulation schedule`, arxiv_id_or_url: `https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html`, url: `https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html`}
  - {name: `TRM long-horizon baseline gate`, arxiv_id_or_url: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM bounded full-fine-tune control`, arxiv_id_or_url: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - principle: Each method/source MUST carry a real arXiv ID or canonical doc URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v383: `lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read the checkpoint-resume, LR-schedule, and long-horizon-training track in
`research-studying.md` and `research-references.md`, including the `.375`
resume-not-restart directive, the `.351` nano-trm PyTorch-Lightning/Hydra
substrate note, and the Exp 4121 `.381` resumable baseline-graft ingestion.
Ran the required helpers:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "checkpoint resume learning rate schedule long horizon training gradient accumulation PyTorch Lightning TRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Tiny Recursive Models checkpoint resume learning rate schedule test time adaptation" --limit 8`

The arXiv cluster helpers emitted reliable verifier, EBM, and active-inference
query URLs. Semantic Scholar returned zero unique arXiv IDs for the TRM
resume query and HTTP 429 for the broader checkpoint/LR query, so it did not
displace the requested anchors. Low-concurrency WebSearch/WebFetch verified
the primary pages for `arXiv:2510.04871`, `arXiv:2511.02886`, the Lightning
checkpoint-resume docs, the Lightning gradient-accumulation docs, and the
PyTorch general checkpoint docs. The `/deep-research` loop was not invoked.

## Current .382 resumable-training anchor

The `.382` headline is narrower than another verifier-search claim. The next
experiment needs to prove that bounded passes are one continuous training
horizon: the same checkpoint identity, same optimizer state, same scheduler
state, monotonic global optimizer step, and accumulated evidence reported as
accumulated-N or accumulated-step, not per-window progress.

For `nano-trm`, this matters because Exp 4108 already showed that a checkpoint
can load and still remain a partial baseline. If the LR schedule rewinds on
each bounded pass, the run can look like a long-horizon TRM reproduction while
actually repeating warmup or cosine decay from a fresh state.

## PyTorch Lightning full-state checkpoint resume

**Method/source:** PyTorch Lightning checkpointing docs
(https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
state that a Lightning checkpoint carries the training state needed to resume,
including epoch, global step, optimizer state, and learning-rate scheduler
state. The same docs say resumed training should call `Trainer.fit(...,
ckpt_path=...)`.

**Implementation over nano-trm + Carnot stack:** Promote this to a gate for the
`.383` training runner. Resume with `ckpt_path`, then assert that checkpoint
source path, global optimizer step, current LR, optimizer param-group LR,
scheduler state, and next logged LR are continuous across the pass boundary.
Persist those receipts in the result JSON before any accuracy comparison.

**Pitfalls / where it fails:** Loading only model weights or re-instantiating a
fresh trainer can make the model look resumed while the optimizer and LR
schedule are new. That hides the exact failure `.382` is supposed to remove.

## PyTorch optimizer-state checkpoint contract

**Method/source:** PyTorch saving/loading docs
(https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
document the general checkpoint shape for resuming training: model state,
optimizer state, epoch or step, and other needed state. Lightning should handle
this for nano-trm, but the contract is still the fallback if a script bypasses
Lightning.

**Implementation over nano-trm + Carnot stack:** Add a minimal checkpoint
receipt verifier that reads the `.ckpt` or `.pt` file and checks for model
state, optimizer state, LR scheduler state when present, global step or epoch,
data split checksum, and RNG/data-cursor receipt. If any are missing, label the
run as a warm-start or mechanism probe rather than continuous training.

**Pitfalls / where it fails:** A model-only checkpoint preserves weights but
loses momentum buffers and parameter-group learning rates. That can change the
optimization trajectory enough to invalidate a long-horizon conclusion even if
validation accuracy improves.

## Lightning gradient-accumulation schedule

**Method/source:** Lightning training-techniques docs
(https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html)
define gradient accumulation as K microbatches producing one optimizer step
and document scheduled accumulation through `GradientAccumulationScheduler`.

**Implementation over nano-trm + Carnot stack:** Report optimizer steps, not
microbatches, as the horizon unit. If gradient accumulation changes across
passes, record the accumulation schedule and effective batch size so LR
schedulers that step per optimizer update are not accidentally compared against
microbatch counts.

**Pitfalls / where it fails:** Treating every microbatch as a training step
overstates horizon length and can advance or evaluate the LR schedule at the
wrong boundary. In distributed training, effective batch size also changes with
device count, so pass-to-pass GPU topology belongs in the receipt.

## TRM long-horizon baseline gate

**Method:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the substrate: a 7M-parameter recursive
model reporting strong Sudoku and ARC-style generalization from a tiny network.

**Implementation over nano-trm + Carnot stack:** Keep the `.381` baseline gate,
but add LR-continuity receipts before training more. The reported baseline
state should be the accumulated global optimizer step, not the current pass
index, and the held-out Sudoku exact-accuracy trace should be keyed to the same
dataset checksum and checkpoint lineage.

**Pitfalls / where it fails:** A technically resumable checkpoint is not a
faithfully reproduced TRM. If LR state resets, a low or flat validation trace
is ambiguous: it could be a real model limit, or it could be a schedule bug.

## TTA-TRM bounded full-fine-tune control

**Method:** TTA-TRM, `arXiv:2511.02886`
(https://arxiv.org/abs/2511.02886), is the adaptation-control anchor because it
reports full fine-tuning within a bounded competition budget, not just LoRA or
task-embedding updates.

**Implementation over nano-trm + Carnot stack:** Keep a no-verifier full
fine-tune arm beside any verifier-admitted arm, with both arms using identical
accumulated optimizer-step budgets and identical scheduler-resume receipts.
The comparison should log checkpoint source, elapsed optimizer steps, LR before
and after resume, and verifier-admission counts.

**Pitfalls / where it fails:** Full fine-tuning can win by adaptation compute
alone. A per-pass LR reset makes this worse because it changes the compute
profile while pretending to preserve the same budget.

## Flagged for the .383 roadmap

`lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383` is the
strongest `.383` candidate. It is the cheapest falsifiable gate before another
verifier or candidate-expansion experiment: resume nano-trm from the latest
checkpoint with `ckpt_path`, prove optimizer/scheduler/global-step continuity
across two bounded passes, and only then spend on per-step verifier work.
"""

STUDYING_SECTION = """## 2026-06-13 Exp 4130 - .382 resumable-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-resumable-training-2026-06-13.md`.

**Filtered track:** checkpoint resume, LR-schedule continuity, and
long-horizon accumulation over the `nano-trm` plus Carnot stack. This follows
the Exp 4121 `.381` baseline-graft ingestion and narrows `.382` to the runner
discipline needed before another verifier-search or training claim.

**Seed and fresh-pass candidates marked ingested:**
- PyTorch Lightning checkpoint resume docs - mapped as the full-state
  `ckpt_path` gate because Lightning checkpoints carry optimizer and LR
  scheduler state as well as global step.
- PyTorch saving/loading docs - mapped as the fallback optimizer-state
  checkpoint contract for any non-Lightning runner.
- Lightning gradient-accumulation docs - mapped as the long-horizon accounting
  rule: count optimizer steps and effective batch size, not microbatches.
- TRM, arXiv:2510.04871 - mapped as the resumed long-horizon baseline whose
  Sudoku evidence must be accumulated by checkpoint lineage and optimizer step.
- TTA-TRM, arXiv:2511.02886 - mapped as the bounded full-fine-tune control that
  must share the same resumed scheduler receipts as any verifier-admitted arm.

Flagged for .383: `lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383`.

**Bottom line for the .383 roadmap:** first ship a Lightning full-state resume
gate for nano-trm that proves optimizer, LR scheduler, global-step, data
checksum, and gradient-accumulation continuity across two bounded passes. If
that gate fails, do not spend the next run on per-step verifier work.
"""

STUDYING_MARKER = "## 2026-06-13 Exp 4130 - .382 resumable-training SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    flagged_for_v383: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4130 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "flagged_for_v383": flagged_for_v383,
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
    if not isinstance(methods_mapped, list) or not 3 <= len(methods_mapped) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

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
        elif source in VERIFIED_CANONICAL_DOC_URLS:
            expected_url = source
        else:
            raise ValueError(
                "method arxiv_id_or_url must be a verified arxiv ID or canonical doc URL: "
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

    flagged = artifact["flagged_for_v383"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v383 must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps resume sources and closes planning."""

    required_phrases = (
        "Current .382 resumable-training anchor",
        "PyTorch Lightning full-state checkpoint resume",
        "PyTorch optimizer-state checkpoint contract",
        "Lightning gradient-accumulation schedule",
        "TRM long-horizon baseline gate",
        "TTA-TRM bounded full-fine-tune control",
        "Implementation over nano-trm + Carnot stack",
        "Pitfalls / where it fails",
        "Flagged for the .383 roadmap",
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
        flagged_for_v383=DEFAULT_FLAGGED_FOR_V383,
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
