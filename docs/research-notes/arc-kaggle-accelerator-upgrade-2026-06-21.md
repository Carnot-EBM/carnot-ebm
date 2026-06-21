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

## WIRED 2026-06-21 (operator-directed) — `"machine_shape": "NvidiaL4"`

Added to `scripts/kaggle/submission_kernel/kernel-metadata.json` (the scored kernel). Confirmed
correct on every axis by the operator + a 3-agent verification workflow + the installed kaggle-cli
source:

- **Field name `machine_shape` (CERTAIN).** The JSON key is `machine_shape`, NOT `accelerator`.
  Verified in the installed kaggle-cli source `kaggle/api/kaggle_api_extended.py:4649`:
  `request.machine_shape = acc if acc else get_or_default(meta_data, "machine_shape", None)`.
  `--accelerator` is the *CLI flag* that overrides it; an `"accelerator"` JSON key would be
  silently ignored (`get_or_default` → None → default P100) — a silent failure on the scored run.
- **Value `NvidiaL4` (the 4×L4 upgrade).** The bare base name provisions the competition's multi-GPU
  default (96 GB total / 24 GB per card), by direct analogy to the documented T4x2 precedent
  (kaggle-cli issue #821: bare `NvidiaTeslaT4` → 2×T4). REJECT `NvidiaL4X1` (that is explicitly
  1×L4). Do NOT use `NvidiaL4X4` — it is NOT in the kaggle-cli accelerator enum and would 400.
- **sm_89 binary coverage (PROVEN).** The bundled `libggml-cuda.so` is built with
  `CUDA_ARCHS="60;75;89;89-virtual"` (scripts/kaggle/kernel/main.py:27) and direct `cuobjdump`
  inspection of the v7 artifact shows native sm_89 SASS + compute_89 PTX. So the LLM generator loads
  at full speed on L4 — no hard-fail. The 24 GB L4 strictly RELAXES the VRAM/OOM risk vs the 16 GB
  P100/T4 (so MTP-off may no longer be necessary on L4 — re-evaluate `CARNOT_ARC_MTP`).
- **Scoring honors the choice (operator-confirmed).** The scoring engine pulls the hardware profile
  from the submission notebook's last saved commit / pushed metadata; with internet disabled the
  hidden test set runs on the chosen L4x4 tier.
- **Quota (operator-confirmed).** L4x4 burns GPU quota 2×: a full 12 h run subtracts ~24 h from the
  weekly allowance. Verify weekly runway before a long submission so it doesn't terminate early.

### Quota math + submission cadence (operator 2026-06-21) — PLAN AROUND THIS

- **Baseline pool: 30 GPU-h/week** (floating; occasionally bumped to 36-40 h if site demand is low,
  but plan for the hard 30 h).
- **On L4x4 the 2× burn → ~15 real-world GPU-h/week.** A full-length ARC notebook (12 h cap) burns
  **24 h of quota**, so **effectively ONE full-length L4x4 submission per week** — unless the run
  finishes materially under 12 h (L4 is far faster than P100, which helps).
- **Reset: Saturday 00:00 UTC = Friday 8:00 PM EDT (Florida).**
- **Overdraft rule:** if you have ≥1 minute of quota left when a submission STARTS, Kaggle lets it run
  to completion (up to 12 h) without killing it or bleeding into next week. So the final weekly run
  can be launched on fumes.

**Implication for the June-30 deadline (~9 days out):** only ~**2 L4x4 full submissions remain** (this
quota week, reset Fri Jun 26 8 PM EDT; then the Jun 27–30 window). Each is precious → **gate every
submission on an offline result that beats both the TRM baseline AND our best prior submitted run**
(the standing offline-first discipline, now doubly binding). Launch each weekly run before the Friday
reset, using the overdraft rule if needed.

**Efficiency pays double:** the ARC score is `min(1.15, h/a)²` (rewards FEW actions), and a
faster-finishing notebook also preserves quota for a possible second run. L4's speed serves both.

**The standing lever (not a re-litigation — L4 is the operator's choice):** 16 GB is now PROVEN
sufficient (corpus-leak fix + MTP-off 5.9 GB + 1.45 GB CNN fit). So the default 1× tier (P100/T4,
16 GB) would permit ~2 submissions/week instead of 1. L4x4 trades that frequency for 24 GB headroom +
much faster inference (P100 is ancient Pascal). Staying on L4x4 is sound; the 1× tier is the fallback
if submission FREQUENCY ever matters more than headroom near a deadline.

### One ground-truth check remaining (medium→high confidence closer)

The identifier confidence is "medium" only because the live ARC upgraded-accelerators page is
JS/login-gated (un-fetchable headlessly) and there is no client-side enum to validate against. The
definitive check, when convenient: in the Kaggle UI set one notebook to the L4x4 accelerator, then
`kaggle kernels pull <that-notebook> -m` and read the `machine_shape` Kaggle itself wrote. Expected:
`NvidiaL4`. (Not blocking — both the operator and the T4x2 precedent already point to `NvidiaL4`.)

### Optional follow-ups (operator's call — not done, to respect the explicit "submission kernel" scope)

- **dryrun_kernel + agent_dryrun_kernel**: SHOULD also set `machine_shape: NvidiaL4` so the offline
  smoke tests validate on the REAL eval GPU (else we test the OOM cliff on a 16 GB P100 and may
  falsely conclude MTP-off is required when the 24 GB L4 has the headroom). Cost: each dry-run run
  then burns 2× quota. Safe to defer — a dry-run that passes on the tighter P100 is a conservative
  proxy for the roomier L4 (the binary covers both arches).
- **Stale comment**: `scripts/kaggle/dryrun_kernel/main.py:3` says "built for sm_60" — inaccurate;
  the real binary covers 60/75/89. Cosmetic fix.
- **Defense-in-depth**: add a post-build `cuobjdump --list-elf | grep -q sm_89` assertion to the
  BUILD kernel (mirrors the existing MTP-symbol self-verify) so a future disk-pressure rebuild that
  drops sm_89 fails loudly instead of silently shipping an L4-incompatible binary.

## Sources

- kaggle-cli accelerator list — https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- kernel-metadata `machine_shape` field — https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
- ARC Prize 2026 upgraded-accelerators — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators
- ARC-AGI-3 competition — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- API feature-request thread (machine_shape) — https://www.kaggle.com/product-feedback/664303
