# Clef system-1 decision API: review and gated spike plan (2026-10-11)

Operator request: try the Ternary Bonsai 2 + MTP + Clef LoRA + image approach, and find ways
to use a fast "system-1" decision API in the live ARC agent.

Sources: the trevest model card (fetched 2026-10-11) and a reply from the `haven` session
(Ian's household stack). I have not run any of it. Every number below is a claim by those two
sources, not a Carnot measurement.

## What it is

| Part | Claim | Source |
|---|---|---|
| Clef | Cloudflare decision model: typed questions over a state, one probability per option, one forward pass | trevest card |
| Weights | Qwen3.8-27B plus a LoRA on layers 40-63. Head, tokenizer and schema code are Cloudflare's, Apache-2.0 | trevest card |
| One model, two APIs | LoRA loads at scale 0 for chat. A decision request sets scale 1 for that request only | haven |
| Engine | llama.cpp commit `ad1de39e0` plus a patch (`/embeddings` takes a per-request LoRA scale) | trevest card |
| Head | CPU sidecar, `POST /v1/systemone`, about 30 ms | trevest card |
| MTP | Separate draft file `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` from `unsloth/Qwen3.8-27B-GGUF` | haven |
| Images, chat | Works with an mmproj file | both |
| Images, Clef | Not in the trevest stack. haven wrote a third patch and tested synthetic images only | haven |

Quality claim (trevest, 500 items each): AG News 91.6, Banking77 89.6, Emotion 50.4. Constrained
decoding with the plain 27B scored 86.2, 77.6 and 49.2.

Speed claims:
- trevest, RTX 5090 Laptop: prefill about 1,100 tok/s, a few-hundred-token decision 0.3 to 0.6 s.
- haven, RTX 3090 Ti: about 0.08 s fixed plus 1.2 ms per prompt token. MTP lifts chat from 78.7 to
  about 110 tok/s with `--spec-draft-n-max 2`.

## What we already have

- Card 1 is free (4 MiB used) and has the same compute capability as haven's card (8.6).
- `unsloth/Qwen3.8-27B-GGUF` Q4_K_M is cached. The MTP file, an mmproj file and the Clef files are not.
- `prism-ml/Ternary-Bonsai-2-27B-gguf` is cached, but as `PTQ1_0`, not the `PQ2_0` file haven used.
  It needs PrismML's llama.cpp fork.

## Decision: skip Bonsai, use the mandated model

The trevest card says any Qwen3.8-27B GGUF works. Our mandate is `unsloth/Qwen3.8-27B-GGUF`, and
the live generator is pinned to it. A Bonsai base would add a second model, a forked engine and
an untested ternary base. It is out of scope until the plain Q4_K_M test passes and VRAM is the
only blocker. Bonsai's value is VRAM, and Kaggle's card has 96 GB.

## Open questions that decide whether this matters

1. Which engine does the scored kernel run? The patch is for llama.cpp only. The shipped
   proposer uses llama-server. Some notes mention vLLM. Read the kernel launch code first.
2. The scored run is offline. Clef files must ship inside the kernel dataset. Size: LoRA 1.3 GB.
3. A LoRA changes the live generator's behavior only if it is applied. At scale 0 it should not.
   That must be measured, not assumed (gate G3).
4. Clef answers typed questions over a text state. ARC state is a grid. We must render it as text
   or as an image. Clef accuracy on images is unverified by anyone.

## Where a fast decision could go in the live agent

These are hypotheses. Each needs a logged decision set to test against. None is built.

| Site | Question | Label source |
|---|---|---|
| `E3AgentPolicy._should_enter_induction` | Is the agent stalled enough to spend a 60,000-token induction call? | Past runs: did the induction call install a plan |
| Untested-action choice (`StepwiseExplorer`) | Which of N candidate actions to try next | Past runs: which action first changed the frame |
| Plan trust | Is the induced world model likely to hold? | `WorldModelVerifier` outcomes |
| Dead-end check | Is this state a dead end | Offline arcade BFS on public games |

Constraint from the live-path rule: only code reachable from `make_carnot_agent` or
`arc_loop_solve.py` counts. A decision module nothing imports is wasted effort.

## Gated plan

Stop at the first failed gate. Write the result to a dated artifact either way.

| Gate | Test | Pass |
|---|---|---|
| G0 | Read the engine patch and `clefhead/` code. No network calls, no writes outside its dirs, no exec of downloaded code | Written audit, nothing hidden |
| G1 | Build llama.cpp at the pinned commit with the patch for sm_86. Serve Q4_K_M, MTP, LoRA at scale 0 and mmproj on card 1 | Server up, VRAM and context recorded |
| G2 | Chat unchanged: same prompts, scale 0 with LoRA loaded vs no LoRA, greedy | Outputs identical or differences explained |
| G3 | The trevest demo requests return the claimed answers and a latency within 2x of the claim | Numbers recorded |
| G4 | Offline decision test on logged live-agent decisions: Clef vs the current rule vs a constant baseline, sample size set before the run | Clef beats the rule on the held-out set. If it does not, stop |
| G5 | Image input to Clef | Only after G4, and only if G4 used text states |

G4 needs a positive control. A null result with a degenerate corpus is not a finding.

## Not in scope

- No change to the frozen live generator or the submission stack.
- No Bonsai build.
- No training. We would use Cloudflare's released adapter as is.

## Cross-references

- `docs/research-notes/limite-1b-violetto-review-2026-10-10.md` (same gated-note format)
- CLAUDE.md: ARC Live-Path Reachability, Energy-Based Calibrated-Decision Training Floor
  (a typed, calibrated decision API is the same capability class)

## Spike results, 2026-10-11 (append-only)

Setup: card 1 (RTX 3090, bus 62:00.0), upstream llama.cpp `ad1de39e0` plus the trevest patch,
built for sm_86 with CUDA 13.4. Base `Qwen3.8-27B-Q4_K_M.gguf`, MTP file, `mmproj-F16.gguf`,
Clef LoRA at scale 0, `-c 16384 -np 1 -ub 2048`, `--spec-draft-n-max 2`. Sidecar on CPU.
Ternary Bonsai was NOT run. Ian authorized it later in the session. It needs PrismML's fork.

| Gate | Result |
|---|---|
| G0 audit | Pass. The patch touches only the embeddings path and the speculative-draft context. The sidecar talks only to the engine URL and reads the GGUF. `safetensors` load, no pickle, no exec |
| G1 serve | Pass. Server up, 22,320 MiB used on a 24 GB card, one card, chat plus LoRA plus MTP plus mmproj |
| G2 chat unchanged | Partial pass. Greedy chat output is identical before and after a Clef call. This shows no state leak. It is NOT a comparison against a server with no LoRA loaded |
| G3 decisions | Pass. 3 of 3 correct on a hand-made ticket test, each repeated twice with identical probabilities. 238 to 242 tokens take 0.41 s on a warm run (first call 0.85 s). Card claim: 0.3 to 0.6 s |
| Chat speed | 63 tok/s on the first short request, 77 tok/s on a longer greedy one, with MTP |

Limits: 3 hand-made tickets are a smoke test, not a quality measure. The card's accuracy table
is still the card's claim. Image input to Clef is untested (G5). Text-only sidecar rejects images.

Next: G4 needs a labeled set of live-agent decisions. See the section below.

### G4 design (what the live agent already has)

The run cells show the agent already carries two small decision components:
`candidate_router` and `frame_change_scorer` (witness keys `candidate_router_loaded`,
`frame_change_scorer_loaded`). The second one failed to load in every induction cell on
2026-10-11 and is still unexamined. So the baseline for G4 is those components, not a constant.

Witness keys named `generator_vllm_*` also show the scored path can use vLLM. The Clef patch is
llama.cpp only. Open question 1 above stays the first thing to settle.

G4 plan, in order:
1. Find why `frame_change_scorer` did not load. If it is a worktree-only fault, Clef has no gap to fill there.
2. Collect labeled decisions from existing run evidence (`results/arc_e3/*/attempts`, read only):
   state before action, candidate actions, which action first changed the frame.
3. Set the sample size before scoring. Compare: Clef on a text rendering of the state, the existing
   router, and a constant baseline. Include a corpus where the best action is not the default one.
4. Pass only if Clef beats the existing router on the held-out set. A tie means no integration.

### Findings, 2026-10-11 (append-only)

1. **The frame-change scorer failure is a worktree artifact.** `load_live_action_effect_scorer`
   returns None when both its inputs are missing: `data/arc_transition_corpus/*.npz` and
   `results/experiment_4629_live_frame_change_cnn.pt`. The main checkout holds both (25 files,
   5.3 MB, and a 14 KB checkpoint). This worktree holds neither. The checkpoint is git-ignored
   (`results/**/*.pt`). So the 2026-10-11 induction run played with the action-effect prior off.
   The scored kernel is not affected by this cause. A rerun must copy both inputs in first.
2. **Open question 1 is answered.** `scripts/kaggle/submission_kernel/main.py` runs the bundled
   `llama-server` plus a GGUF by default. vLLM is chosen only when the NVFP4 safetensors dataset is
   attached. So the llama.cpp patch applies to the default path. It still needs a rebuilt
   server binary for the Kaggle card (not sm_86) and the 1.3 GB LoRA shipped in the kernel dataset.
   The vLLM path would have no Clef support.
3. **G4 baseline.** The existing router and scorer need their inputs present. G4 must run where
   they load, or the baseline is a broken component and any Clef win means nothing.
