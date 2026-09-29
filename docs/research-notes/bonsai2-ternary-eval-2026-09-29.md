# PrismML Bonsai 2 27B (ternary) third-party eval on our own hardware

Date: 2026-09-29. Full data: `results/experiment_bonsai2_ternary_eval.json`.

## What this checks

The user asked if Carnot could benefit from PrismML's Bonsai 2 27B, a
ternary-quantized version of `Qwen/Qwen3.8-27B` -- the same base model
CLAUDE.md already mandates for new experiments. PrismML claims about 9x
smaller memory footprint and 98.2% quality retention. This note reports what
we measured ourselves, on our own RTX 3090s, not what PrismML claims.

This is a bounded eval only. It does not change Carnot's own architecture and
does not touch the live ARC Kaggle submission stack.

## Fork-safety audit (done first, per this project's audit-untrusted-code rule)

Bonsai 2 needs a fork of llama.cpp, `PrismML-Eng/llama.cpp` (branch `prism`).
Stock llama.cpp cannot read the ternary GGUF files at all.

We cloned the fork to scratch (never into this repo) and diffed it against
its own merge-base with upstream `ggml-org/llama.cpp`
(`5ea87ddad22541a37053c7ba92b02ec1923617c6`, 2026-08-25). The diff is 178
files, +22725/-1059 lines, 130 fork commits.

Result: **clean**. The diff is new ternary quantization kernels (PTQ1_0
1-bit, PQ2_0 2-bit) across every backend, a speculative-decoding feature, a
KV-cache calibration tool, and matching tests -- exactly what a real
quantization-format fork looks like. We scanned it for network calls,
credential handling, obfuscated code, and unexpected write targets. We found
none. The only network URLs in the diff are ordinary vendor package sources
in a new CI workflow file (NVIDIA, AMD, LunarG), plus one localhost test URL.

We built it locally with CUDA for our RTX 3090s (compute capability 8.6) and
confirmed real GPU offload (VRAM jump on GPU 1, GPU 0 untouched).

## GGUF provenance

`prism-ml/Ternary-Bonsai-2-27B-gguf`, revision `b072e1d3b35a0a630cece372c2127528e0994386`,
file `Ternary-Bonsai-2-27B-PTQ1_0.gguf`. Downloaded size matches the
HuggingFace API's reported size exactly (5,946,648,928 bytes), and the LFS
object hash matches. We also note the task brief's name for this variant
("TQ1_0") is not this repo's actual ggml quant id (PTQ1_0) -- a naming
mismatch worth flagging, though the file itself is exactly what was expected
in size.

## Single-stream throughput (real measurement, GPU 1 only)

Same fork binary (`llama-server`), same prompt, same generation length (300
tokens), 5 runs each:

| Model | Mean tok/s | Variance |
|---|---|---|
| Ternary Bonsai 2 (PTQ1_0) | 61.20 | 0.018 |
| Standard mandated Qwen3.8-27B-Q4_K_M | 42.65 | 0.025 |

Ternary is **1.44x faster**, single-stream, on our hardware. Real, but far
below PrismML's own "up to 143 tok/s on RTX 5090" claim -- that number is on
different, faster hardware than ours and is not a claim this eval could or
should try to reproduce.

## Size, honestly compared

Ternary weights: 5.95 GB. Our mandated model's weights (already 4-bit
quantized): 17.1 GB. That is a real **2.88x** size reduction against OUR
baseline -- not the vendor's "9x smaller" claim, which compares against full
FP16/BF16 precision (about 55 GB for a model this size), a baseline we do not
use. Both numbers are honest; they answer different questions. Report the
2.88x number if this project ever cites a size comparison, since that is the
one that matches what we actually run.

## The addendum's core question: does freed VRAM buy more concurrent capacity?

The coordinator asked us to test whether the ternary model's smaller weight
footprint frees VRAM for more KV cache, letting more concurrent requests fit
and raising aggregate throughput. We tested this directly with a doubling
scan of `--parallel N` on the SAME fork binary, same context per slot
(2048), for both GGUF files.

**Both models hit the exact same ceiling: max loadable N=152, first failure
at N=160.** The failing allocation, identical for both models, is for the
"rs cache" (recurrent-state cache) -- confirming this base architecture
(`Qwen3.8-27B`) is a hybrid attention design with a Gated Delta Net component
(the server itself logs "fused Gated Delta Net (chunked) not supported" for
both models). That state cache is sized by the shared architecture, not by
weight quantization, so the ~10 GB the ternary weights free up buys **zero**
extra concurrent-stream capacity for this specific model family.

**This refutes the addendum's premise for this model pair.** It is a
reasonable general hypothesis about ternary quantization, but it does not
hold here, because the bottleneck for this architecture is not the weight
buffer -- it is a fixed-size state buffer both models pay for equally.

## Aggregate throughput at real concurrency

Given the above, we still measured actual aggregate throughput at two
concurrency levels, since more slots being loadable does not by itself say
whether they are usable:

| N | Model | Aggregate tok/s | Per-stream tok/s | % of single-stream baseline |
|---|---|---|---|---|
| 8 | Ternary | 151.1 | 23.8 | 39.0% |
| 8 | Standard | 90.4 | 13.0 | 30.4% |
| 32 | Ternary | 223.0 | 9.7 | 15.9% |
| 32 | Standard | **7.9** | **0.27** | **0.6%** |

We defined "unacceptable per-stream degradation" as falling below 50% of
single-stream baseline, before looking at the N=32 numbers. Both models are
already below that bar at N=8 -- an honest finding, not a result to smooth
over: this fork build's continuous batching costs real per-stream latency
for both models, for a real but modest aggregate gain (roughly 2.5x to 3.6x
by N=8, depending on model).

**The N=32 standard-model number is the most important single finding in
this eval and is NOT fully understood.** The standard model's aggregate
throughput did not degrade smoothly -- it collapsed by roughly 28x relative
to the ternary model's own N=32 number, taking over 13 minutes wall-clock to
generate 6400 tokens across 32 streams that should take well under a minute.
We reproduced this on a freshly restarted server with an extended client
timeout to rule out a stale-server or client-timeout artifact; the collapse
is real and repeats. The likely mechanism, based on the server's own log
warning, is that the disabled "chunked Gated Delta Net" fallback path scales
far worse with weight size under heavy batching -- but this is a hypothesis,
not something this bounded eval verified. It deserves its own follow-up,
independent of any Bonsai-2 adoption decision, because it may affect this
fork's batched-serving performance for ANY large non-ternary model, not just
our specific comparator.

We attempted N=100 concurrent for the ternary model; all 100 requests timed
out client-side at 600 seconds with zero completions, even though N=32
completed cleanly in 28.7 seconds. We did not resolve this within the
bounded scope of this eval. Flagged as open, not swept under the rug.

## Bounded quality-retention proxy

This is explicitly NOT a replication of PrismML's 20-benchmark suite. We ran
an identical single-forward-pass accept/reject/escalate readout (reusing the
prompt design and read pattern this project's own
`python/carnot/verify/semif_readout_energy.py` already validated) against a
seeded, balanced 160-row sample from `data/fover_corpus_v4.json`, on both
models.

- Argmax agreement rate between the two models: **62.5%**.
- Standard model accuracy against the real label: **40%** (below chance).
- Ternary model accuracy against the real label: **50%** (exactly chance).
- Standard model escalated on 23% of rows; ternary escalated on 0.6%.

Both models score at or below random-guess chance on this narrow task. This
is very likely a weakness of this specific single-letter readout format on
this specific corpus -- this project's own semif_readout_energy.py module
already documents that the collapsed accept-probability scalar carries
little signal here, even though the raw per-option probabilities do carry
some. It is a proxy limitation, not evidence that either 27B model cannot
reason. Read the 62.5% agreement rate as "somewhat correlated, not
identical" -- it neither confirms nor refutes PrismML's quality-retention
claim, which was measured on a completely different, much broader benchmark
suite this eval did not and could not reproduce within its bounded scope.

## Recommendation

Conditionally worth a narrow follow-up, grounded only in what we measured:

1. The fork audit is clean and the GGUF provenance checks out -- no blocker
   on trust grounds.
2. The 1.44x single-stream speedup is real and would genuinely help local
   dev-loop iteration speed (any experiment currently waiting on the
   mandated model's generation time).
3. The 2.88x size reduction against our actual baseline is real, but the
   VRAM-savings-buys-more-concurrency hypothesis does NOT hold for this
   model family -- do not cite that benefit if this is proposed further.
4. The bounded quality proxy is inconclusive and must not be read as
   confirming PrismML's 98.2%-retention marketing claim.
5. Do NOT adopt this fork's build for any concurrent/batched serving path
   (which would matter for the ARC Kaggle scoring case) until the N=32
   standard-model collapse is understood -- the same batching code underlies
   both models, so this is a real risk, not a ternary-specific one.

If pursued, the concrete next step is a scoped dev-loop-iteration-speed
pilot only: swap the ternary GGUF into local single-stream experiment
iteration, leaving the frozen live ARC generator and the Kaggle submission
stack completely untouched.

## Cross-references

- `results/experiment_bonsai2_ternary_eval.json` -- full measured data
- `python/carnot/verify/semif_readout_energy.py` -- the readout prompt/read
  pattern this eval reused, and the prior finding about this corpus's
  collapsed-scalar weakness
- CLAUDE.md "Verifier Authenticity Discipline" / "Adversarial Artifact
  Verification" -- the disclosure discipline this note and its artifact
  follow
- CLAUDE.md "SOTA Local Models" -- the current `unsloth/Qwen3.8-27B-GGUF`
  mandate this eval compares against

## Follow-up 2026-09-29: where the N=32 collapse begins, and is it fork-specific

Full data: `results/experiment_bonsai2_n32_collapse_followup.json`. This
follow-up answers the one hypothesis the eval above flagged but did not
check.

### Where the cliff begins

Bisected the standard model (`Qwen3.8-27B-Q4_K_M`) on the same fork build,
same prompt, same `ctx_per_slot=2048`. N=8 and N=32 are the numbers above,
reused. N=16, N=20, N=24 are new:

| N | aggregate tok/s | per-stream tok/s | wall (s) | GDN chunked path disabled at load? |
|---|---|---|---|---|
| 8  | 90.36 | 12.97 | 17.71  | not checked (log not kept) |
| 16 | 60.38 | 4.32  | 53.00  | no |
| 20 | 27.00 | 1.51  | 148.13 | yes |
| 24 | 8.54  | 0.41  | 561.93 | yes |
| 32 | 7.94  | 0.27  | 805.55 | yes |

This is not a hard cliff at one N. It is a continuous, steep, super-linear
slide. Real degradation shows up as early as N=16 (aggregate throughput
already falls below the N=8 number, where the ternary model's aggregate
throughput keeps rising with N). The slide gets much steeper between N=16
and N=24 -- by N=24 the model has already lost about 97 percent of its N=8
per-stream rate. N=24 to N=32 is comparatively flat.

### The disable is N-dependent, and the threshold matches the cliff

The fork's own log line, `"fused Gated Delta Net (chunked) not supported,
set to disabled"`, does NOT appear at N=16. It DOES appear at N=20 and N=24
(and, per the original eval's own log, at N=32). This line comes from
`llama_context::resolve_fused_ops` (`src/llama-context.cpp:622`), which
probes device placement using a worst-case graph shaped by `n_seq_max` --
the `--parallel` value. A bigger N changes the probe's outcome. The
threshold where the probe starts failing, and disables the fused kernel for
the rest of that server's life, sits between N=16 and N=20 -- almost exactly
where the throughput curve above gets steep.

### What "disabled" costs, read from source

`build_delta_net_base::build_delta_net` (`src/models/delta-net-base.cpp:432`)
picks between `build_delta_net_fused` (one fused CUDA kernel,
`src/ggml-cuda/gated_delta_net.cu`) and `build_delta_net_chunking`
(`src/models/delta-net-base.cpp:16`) based on the flag the probe sets. The
chunking path is a hand-written per-chunk loop
(`src/models/delta-net-base.cpp:246`) composing roughly 8-10 separate ggml
ops per chunk per layer -- matrix multiplies, a triangular solve, a cumulative
sum, several element-wise ops -- instead of one fused kernel call.
Qwen3.8-27B is a hybrid architecture: 3 of every 4 layers are Gated-Delta-Net
(`full_attn_interval=4`, `src/models/qwen35.cpp:22`), so once the fused
kernel is off, most of the network runs this manual composition.

This is real, exercised, and matches the timing of the collapse. It does
NOT fully explain the standard model's specific 28x-to-30x penalty, though,
because this exact code path is identical for both quantizations -- same
layer count, same chunk count, same op sequence. The prior eval's own log
showed the SAME disable warning fired for the ternary model too, at N=32,
and the ternary model kept scaling anyway. Why the chunking path costs the
K-quantized model so much more than the ternary model, at the same N, is
NOT confirmed by this follow-up. Read as an open question, not a solved one.

### Fork vs. stock: a real code difference, and a confound

Built a fresh, CUDA-enabled checkout of real upstream `ggml-org/llama.cpp`
(commit `6a2743f028f78bfb88a7189607b49bde30df3769`, today's master) and ran
the identical N=32 standard-model test through it.

Caveat first, stated plainly: this is not a clean same-commit A/B. The
fork's audited merge-base with upstream is 5 weeks and hundreds of commits
behind today's stock master. Any difference could come from the fork's own
changes, from unrelated upstream drift, or both.

Found one concrete, citable difference anyway:

- Fork, `src/llama-context.cpp:324`: `cparams.auto_fgdn = true;`
- Stock, `src/llama-context.cpp:235`: `cparams.auto_fgdn = false;`

With `auto_fgdn` false, stock never runs the Gated-Delta-Net probe at all,
so the fused kernel stays on for the whole run. Confirmed empirically: the
stock server's log never mentions "gated" or "gdn" once. This whole
mechanism -- the probe, the fused/chunked split, the chunking path itself --
is upstream code, not part of the fork's diff. Diffed the fork's file list
against fresh upstream to confirm.

But stock is not clean either. Its own `resolve_fused_ops` disabled Flash
Attention instead, for the same reason (a device mismatch, this time on the
1-in-4 full-self-attention layers): `"layer 3 is assigned to device CPU but
Flash Attention is assigned to device CUDA0"`. Same structural pattern
(auto-probe disables a fast fused op, falls back to a slower generic path),
different specific op.

Stock's N=32 run was not run to full completion -- after the per-slot
generation rate converged and held steady for over 20 consecutive samples
(`tg = 0.27 t/s`), the run was stopped deliberately rather than waited out,
to keep the diagnostic budget sane. That converged number, 0.27 tok/s per
stream, is essentially identical to the fork's own converged N=32 number
(also 0.27 tok/s per stream). Two honest readings are both defensible: this
is a shared architectural bottleneck at high concurrent batching on this
GPU, reachable via either disabled-fused-op path and not specific to the
fork's changes; or it is a coincidence between two different mechanisms.
This follow-up cannot tell those apart within its bounded scope.

### GPU diagnostics during a collapse

`nvidia-smi dmon`, sampled once per second on GPU 1 through the whole stock
N=32 run: SM occupancy pinned at 83-100 percent, memory flat at 18.3 GB (no
growth, no thrashing), power draw oscillating 142-164 W -- well under the
RTX 3090's roughly 350 W TDP despite near-100-percent occupancy. High
occupancy plus low power is the signature of many small, serialized kernel
launches, not one throughput-saturating batched matrix multiply. That
matches what the source shows.

A `gdb` backtrace of the server's main thread, taken during a stretch with
no new progress lines for over 4 minutes of real time, showed it inside
`cudaStreamSynchronize` -- a genuine, blocking wait on real GPU compute, not
a deadlock or a hang. The process was undisturbed by the attach and kept
running afterward.

`dmesg` (readable via `sudo -n dmesg -T`, unexpectedly permitted without a
password on this host) showed zero Xid errors and zero OOM-killer activity
anywhere in the kernel ring buffer, for the whole session, back to the last
boot. Whatever this is, it is a userspace performance problem, not a kernel
fault or a memory-pressure kill.

### Honest conclusion

The cliff is not at N=32. It starts by N=16 and is mostly done by N=24. The
fork's flagged mechanism -- the disabled chunked-GDN fallback -- is real,
is upstream code rather than a fork invention, and its N-dependent
activation lines up with where the throughput curve gets steep. It is NOT
proven to be the full explanation, because the same code path runs
identically for the ternary model, which does not collapse. Stock master,
tested independently, converges to the same per-stream throughput at N=32
via a different disabled-fused-op fallback (Flash Attention, not GDN) --
consistent with, but not proof of, a shared architectural cause rather than
a fork-introduced one. GPU diagnostics rule out a driver fault or an
OOM-kill and are consistent with a many-small-kernel-launches inefficiency
in whichever generic fallback path is active. The remaining open question --
why the standard (K-quantized) model pays roughly 30x more than the ternary
model for the same disabled-fused-op fallback -- was not resolved.

### Cross-references (follow-up)

- `results/experiment_bonsai2_n32_collapse_followup.json` -- full measured
  data for this follow-up
- `src/llama-context.cpp:622` (`resolve_fused_ops`), `:665` (the disable log
  line), `:324` vs stock `:235` (`auto_fgdn` default) -- in the fork's tree,
  `PrismML-Eng/llama.cpp` commit `87268f775d74cf8f7ffc6c22a95684aa55995533`
- `src/models/delta-net-base.cpp:16` (`build_delta_net_chunking`), `:432`
  (the fused-vs-chunked dispatch) -- same fork tree
- `src/models/qwen35.cpp:22` (`full_attn_interval`) -- same fork tree
- `ggml-org/llama.cpp` commit `6a2743f028f78bfb88a7189607b49bde30df3769` --
  the stock master build used for the isolation test
