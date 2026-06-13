# SOTA ingestion 2026-06-13: resumable training with LR-schedule continuity

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

