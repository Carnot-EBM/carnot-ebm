# Kaggle notebook accelerator upgrade for the ARC-AGI-3 submission (2026-06-21 outer-loop)

Operator question: "I'm reading that it may be possible to upgrade our notebook hardware backend,
can you see an API option to do that?" Answer: **yes — via the `machine_shape` field in
`kernel-metadata.json`, and the ARC Prize 2026 competitions explicitly offer upgraded accelerators
(L4x4, 24 GB/GPU) to participants.** This directly relaxes the dual-16 GB squeeze we have been
engineering against.

## The API knob

`kernel-metadata.json` accepts an optional **`machine_shape`** field (equivalently the
`kaggle kernels push --accelerator <id>` CLI flag). Valid GPU identifiers as of the Feb-2026
kaggle-cli docs:

`NvidiaTeslaP100` (default), `NvidiaTeslaT4`, `NvidiaTeslaT4Highmem`, `NvidiaL4`, `NvidiaL4X1`,
`NvidiaA100`, `NvidiaH100`, `NvidiaRtxPro6000`, plus TPU variants.

Critical caveat from the docs: *"some of these are only available to participants of specific
competitions, and some are only available to Kaggle admins."* So the high-end IDs (A100/H100/L4x4)
are gated — availability is competition- and account-scoped, not universal.

## ARC Prize 2026 specifically offers the upgrade

- The ARC Prize 2026 competitions have a dedicated **"upgraded-accelerators"** overview page
  (`/competitions/arc-prize-2026-arc-agi-3/overview/upgraded-accelerators`).
- Kaggle's own description: *"notebooks with L4x4s consume GPU quota at twice the rate of the older
  T4x2 and P100 machines"* — confirming **L4x4 is available to ARC Prize 2026 participants**, at a
  2× GPU-quota cost.
- **L4 = 24 GB VRAM** vs P100/T4 = 16 GB. Per our own
  `arc-agi3-kaggle-submission-requirements-2026-06-17.md` (line 179): *"T4 16 GB vs L4 24 GB. At
  24 GB the 16 GB constraint relaxes."*

## What this changes for our submission

We have been engineering hard against a **dual 16 GB ceiling** (VRAM AND ~16 GB system RAM, both
shared with Qwen3.5-9B): MTP-off to drop the LLM 11.8→5.9 GB VRAM, the `CARNOT_ARC_NGL`
prefill-to-RAM lever, the corpus-load RAM-leak fix (7.5 GB→0.18 GB/game). On an **L4x4** most of
that pressure evaporates: 24 GB VRAM/card + (typically) more host RAM means the LLM + the live CNN
fit + KV-cache fit with wide headroom, and the prefill-to-RAM trade-off (which moves pressure from
VRAM to the equally-scarce RAM) becomes unnecessary.

Those fixes are NOT wasted — the corpus leak and the npz-decompression bug were correctness defects
that would bite at any memory size; they just stop being load-bearing.

## Current state + what to verify before flipping it

- **Our kernels currently request NO `machine_shape`** — every `kernel-metadata.json` has only
  `"enable_gpu": true`, which yields the DEFAULT (P100, 16 GB). We are not requesting the upgrade.
- To upgrade the scored submission, add `"machine_shape": "<L4 id>"` to
  `scripts/kaggle/submission_kernel/kernel-metadata.json`.
- **Three things to confirm in the live Kaggle UI / competition rules before relying on it** (the
  competition page is JS-rendered; WebFetch couldn't read the dynamic content):
  1. The **exact identifier** for the 4×L4 config. The CLI docs list `NvidiaL4` and `NvidiaL4X1`;
     the competition calls it "L4x4". Confirm whether `NvidiaL4` IS the x4 config or there is a
     distinct x4 string, by selecting it once in the notebook UI and reading back the metadata.
  2. **Scoring fidelity** — ARC submissions re-run YOUR notebook, so the accelerator set on the
     notebook should be what scoring uses (unlike pinned code-competition envs). Confirm on the
     competition's submission-rules page.
  3. **Quota math** — L4x4 burns GPU quota 2×; a 12 h submission run is a large quota draw. Check
     the weekly GPU-hour budget against planned submission count before the June-30 milestone.

## Sources

- kaggle-cli accelerator list — https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- kernel-metadata `machine_shape` field — https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- ARC Prize 2026 upgraded-accelerators — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators
- ARC-AGI-3 competition — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- API feature-request thread (machine_shape) — https://www.kaggle.com/product-feedback/664303
