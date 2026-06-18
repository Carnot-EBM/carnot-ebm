# Model alternatives for the 16 GB Kaggle ARC budget — shortlist to benchmark (2026-06-18)

Operator-requested: find open-weight models worth benchmarking that also fill the need behind
[[project_kaggle_16gb_gemma12b]] — fit **16 GB VRAM with KV-cache headroom** (Q4_K_M weights ≤ ~8–10 GB),
strong at **predicate generation + grid reasoning**, **permissively licensed**, GGUF/llama.cpp, and ideally
with **controllable chain-of-thought** (the forced-thinking-floods-the-canvas failure we hit on
DiffusionGemma). Baseline to beat: **gemma-4-12B-it** (6.7 GB Q4_K_M, ~9 GB KV headroom).

> Method note: the `/deep-research` fan-out workflow rate-limited out (server-side 429s, 0 sources — the
> known 2026-06-11 failure mode). Recovered via the reliable **sequential WebSearch** channel. Numbers
> below are from public benchmark roundups + HF cards (cited); treat self-reported coder scores as
> indicative until we run our own Layer-B harness.

## Ranked shortlist (download + benchmark these)

**1. Qwen3-14B (or the newer Qwen3.5-9B) — TOP PICK.**
Apache-2.0. The decisive feature is **hybrid thinking with seamless on/off switching** (`enable_thinking`
/ `/no_think`) — this DIRECTLY fixes the failure we hit (forced CoT flooding fixed output): thinking-ON
for hard rule induction, thinking-OFF for clean direct predicate emission, one model. Strong code +
reasoning. Q4_K_M ≈ 9 GB (Qwen3-14B) or ≈ 5.5 GB (Qwen3.5-9B → even more KV headroom). Qwen3 dense sizes
0.6/1.7/4/8/14/32B; Qwen3.5 adds 4B/9B. GGUF from Qwen + unsloth. *Why it could beat gemma-4-12B:* the
thinking toggle is the single best fit for our generate-checkable-code need, and it's a stronger reasoner.

**2. Qwen2.5-Coder-14B-Instruct — best dedicated CODER (the #1 need).**
Apache-2.0, dense (MoE-free, simple/fast inference), GGUF (unsloth + official, incl. 128K). Best-in-class
small coder for writing checkable Python predicates / structured output. Q4_K_M ≈ 9 GB. No forced
thinking. *Why:* predicate generation is the load-bearing Layer-B job and a dedicated coder should ground
more rules / fewer malformed outputs than the general gemma-4-12B.

**3. Phi-4 (14B) — best small REASONER, MIT.**
MIT license (cleanest of all). **MATH 80.4%** (beats 3×-larger models; vs Qwen2.5-14B 75.6%, Llama-3.3-8B
68.0%), runs at Q4_K_M in ~8 GB. No forced thinking, strong structured output + instruction-following.
*Why:* the grid-rule reasoning half — Phi-4 punches far above its size on math/logic, which maps to the
abstract rule induction behind a win-predicate.

**Honorable mentions (fit, worth a look):**
- **DeepSeek-Coder-V2-Lite-16B-A2.4B** — HumanEval **81.1%**, MBPP+ 68.8%, Q4_K_M **10.36 GB** (fits 16 GB,
  ~5 GB KV headroom), MoE so fast (2.4B active). DeepSeek custom license (permissive for personal/small
  use — flag, not Apache). Strongest coder benchmark in range.
- **Qwen3-8B** — Apache, controllable thinking, Q4 ≈ 5 GB (most KV headroom) — the budget-friendly Qwen3.
- **Mistral-Nemo-12B** — Apache-2.0, Q4 ≈ 7 GB, solid generalist, no thinking.

## On the grid-reasoning half (architecture note, not a single-model pick)

No general ≤14B LLM is expected to *dominate* abstract grid reasoning at Q4 — that's the project's
existing **hybrid** thesis ([[feedback_hybrid_pragmatic_architecture]]): a small general LLM/coder for
predicate generation + the **TRM** (tiny recursive model, ~7M — already in-repo) / energy verifier for the
grid-reasoning + verification. So the right move is to swap the LLM *generator* slot (gemma-4-12B → a
better-fitting coder/reasoner) and keep TRM + the energy verifier for grid + selection. A dedicated coder
(pick 2) + TRM beats one general model trying to do both.

## Swap-in effort (all three top picks)

All are GGUF/llama.cpp-ready and slot into the existing `cached_sota_pair` / llama-server path — same as
gemma-4-12B. Gotchas: (1) **Qwen3/3.5** — set the thinking mode explicitly (`/no_think` in the prompt or
`enable_thinking=false` in the chat template) so it emits code directly; verify the embedded GGUF
tokenizer per the GGUF-tokenizer rule (load via `.gguf` path, never `AutoTokenizer` on the `-GGUF` repo).
(2) **Qwen2.5-Coder-14B** — standard Qwen chat template, no thinking knob needed. (3) **Phi-4** — MIT,
standard chat template, no thinking knob. All fit 16 GB with KV headroom; Qwen3-8B / Qwen3.5-9B leave the
most room for the long ARC-harness contexts.

## Recommendation

Download and benchmark **Qwen3.5-9B (or Qwen3-14B), Qwen2.5-Coder-14B, and Phi-4** on the exact Layer-B
grounding harness (ka59/tn36, same as the gemma/Qwopus/DiffusionGemma runs) on the iGPU. The standout
hypothesis: **Qwen3/3.5's thinking toggle** gives us gemma-4-12B-class fit + a stronger reasoner + the
on-demand-CoT control that none of the others have — the most likely single upgrade for the generator slot
while staying inside the 16 GB budget.

Sources: [Best Local LLM 2026 (RockB)](https://baeseokjae.github.io/posts/best-local-llm-models-2026/) ·
[AceCloud open-source LLMs 2026](https://acecloud.ai/blog/best-open-source-llms/) ·
[HF blog: open-source LLMs 2026](https://huggingface.co/blog/daya-shankar/open-source-llms) ·
[Qwen3 (Apache, hybrid thinking)](https://qwen-3.com/en) ·
[Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) ·
[Qwen2.5-Coder-14B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF) ·
[Best local coding LLM 2026 (Qwen2.5-Coder vs DeepSeek-Coder-V2 vs Codestral)](https://dev.to/jovan_chan_9500711396d4e6/best-local-coding-llm-in-2026-qwen25-coder-vs-deepseek-coder-v2-vs-codestral-45g8) ·
[DeepSeek-Coder-V2 (Open Laboratory)](https://openlaboratory.com/models/deepseek-coder-v2/)
