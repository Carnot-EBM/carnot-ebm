#!/usr/bin/env python3
"""PHASE 1 -- the n_ctx / max_tokens / VRAM arithmetic, computed from the repo's OWN constants.

Nothing here is a hand-typed number except the card size and the MEASURED ft09 prompt length.
Every coefficient is imported from `arc_executable_world_model`, so if a constant moves this
table moves with it instead of quietly going stale -- which is the same discipline
`_generator_cuda_min_free_mb()` applies to the guard it computes.

THE TWO INEQUALITIES THAT HAVE TO HOLD SIMULTANEOUSLY

  (1) POOL ADMISSION      n_ctx >= K * (prompt_tokens + max_tokens)
      llama-server runs 4 kv_unified slots sharing ONE pool of `-c` cells, so K is the number
      of induce requests in flight. K=4 is the eval-framework shape (one thread per game);
      K=1 is the single-game diagnostic shape every measurement in this session used.

  (2) VRAM FIT            18940.7 + 0.050293*n_ctx + 206.83*slots - 195.3*L + margin <= free
      the measured gemma-4-31B-it Q4_K_M mtp-off envelope. `margin` is the shipped 1500 MiB
      guard slack. L is CPU-offloaded FFN layers (`CARNOT_ARC_FFN_CPU_LAYERS`).

They pull in opposite directions: raising `max_tokens` raises the pool that (1) demands,
which raises the VRAM that (2) has to find. That coupling is the whole content of Phase 1's
"note the coupling" instruction, and it is why a budget raise is not free.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

CARD_TOTAL_MIB = 24576  # RTX 3090
CARD_FREE_IDLE_MIB = 24123  # observed on both idle cards, 2026-07-30
# MEASURED through the generator's own tokenizer (llama_cpp vocab_only on the .gguf path),
# on the prompt the live ft09 run actually built at its induction point (25 transitions,
# 64x64 grid, k=8, no playbook exemplars, code-only directive + fence included).
FT09_PROMPT_TOKENS = 4343


def predicted_vram(n_ctx: int, layers: int = 0, mtp: bool = False, slots: int = 4) -> float:
    v = (
        e3._VRAM_GEMMA31B_INTERCEPT_MIB
        + e3._VRAM_GEMMA31B_PER_CTX_MIB * n_ctx
        + e3._VRAM_PER_SLOT_MIB * slots
        - e3._VRAM_PER_CPU_FFN_LAYER_MIB * layers
    )
    if mtp:
        v += e3._VRAM_MTP_HEAD_INTERCEPT_MIB + e3._VRAM_MTP_HEAD_PER_CTX_MIB * n_ctx
    return v


def derived_n_ctx(max_tokens: int) -> int:
    """What `_default_induce_n_ctx()` returns for a given completion budget."""
    need = e3._LLAMA_SERVER_DEFAULT_SLOTS * (
        e3._INDUCE_WORST_CASE_PROMPT_TOKENS + max_tokens
    )
    return int(-(-need // 4096) * 4096)


print("CONSTANTS READ FROM THE REPO")
print(f"  _INDUCE_DEFAULT_MAX_TOKENS        = {e3._INDUCE_DEFAULT_MAX_TOKENS}")
print(f"  _INDUCE_WORST_CASE_PROMPT_TOKENS  = {e3._INDUCE_WORST_CASE_PROMPT_TOKENS}")
print(f"  _LLAMA_SERVER_DEFAULT_SLOTS       = {e3._LLAMA_SERVER_DEFAULT_SLOTS}")
print(f"  gemma31b envelope                 = {e3._VRAM_GEMMA31B_INTERCEPT_MIB}"
      f" + {e3._VRAM_GEMMA31B_PER_CTX_MIB}*n_ctx + {e3._VRAM_PER_SLOT_MIB}*slots")
print(f"  per CPU-FFN layer credit          = {e3._VRAM_PER_CPU_FFN_LAYER_MIB} MiB")
print(f"  guard margin                      = {e3._GENERATOR_CUDA_GUARD_MARGIN_MIB} MiB")
print(f"  ft09 REAL induce prompt (measured)= {FT09_PROMPT_TOKENS} tokens")
print()

print("A. THE SHIPPED DEFAULT DERIVATION (K=4 worst-case prompt), and its 24 GiB fit")
print(f"{'max_tokens':>11} {'-> n_ctx':>10} {'VRAM L=0':>10} {'+margin':>9} "
      f"{'fits 24123?':>12} {'L needed':>9}")
for mt in (4096, 6144, 8192, 12288, 16384, 24576, 32768):
    n = derived_n_ctx(mt)
    v = predicted_vram(n, 0)
    need = v + e3._GENERATOR_CUDA_GUARD_MARGIN_MIB
    fits = need <= CARD_FREE_IDLE_MIB
    over = need - CARD_FREE_IDLE_MIB
    lay = 0 if fits else int(-(-over // e3._VRAM_PER_CPU_FFN_LAYER_MIB))
    print(f"{mt:>11} {n:>10} {v:>10.0f} {need:>9.0f} {str(fits):>12} {lay:>9}")
print("  -> the SHIPPED default (4096) ALREADY does not fit a 24 GiB card: this is why every")
print("     local measurement pins CARNOT_ARC_INDUCE_N_CTX=32768 rather than taking the default.")
print()

print("B. WHAT A 24 GiB CARD CAN HOLD (L=0, mtp off, 4 slots, 24123 MiB free)")
lo, hi = 4096, 262144
best = 0
for n in range(4096, 262145, 4096):
    if predicted_vram(n, 0) + e3._GENERATOR_CUDA_GUARD_MARGIN_MIB <= CARD_FREE_IDLE_MIB:
        best = n
print(f"  max n_ctx (4096-multiple), L=0 : {best}  "
      f"(VRAM {predicted_vram(best,0):.0f} + {e3._GENERATOR_CUDA_GUARD_MARGIN_MIB} margin)")
for L in (8, 16, 24, 32, 40):
    b = 0
    for n in range(4096, 262145, 4096):
        if predicted_vram(n, L) + e3._GENERATOR_CUDA_GUARD_MARGIN_MIB <= CARD_FREE_IDLE_MIB:
            b = n
    print(f"  max n_ctx (4096-multiple), L={L:<2}: {b}")
print()

print("C. WHAT THE PINNED n_ctx=32768 POOL ADMITS, against ft09's REAL 4343-token prompt")
print(f"{'K':>3} {'per-slot cells':>15} {'max_tokens ceiling':>19} {'4096 ok?':>9} "
      f"{'8192 ok?':>9} {'16384 ok?':>10}")
for K in (1, 2, 4):
    per = 32768 // K
    ceil_ = per - FT09_PROMPT_TOKENS
    print(f"{K:>3} {per:>15} {ceil_:>19} "
          f"{str(ceil_ >= 4096):>9} {str(ceil_ >= 8192):>9} {str(ceil_ >= 16384):>10}")
print("  -> at K=1 (the shape every cell in this session ran) a 32768 pool admits a completion")
print("     budget up to 28425 tokens for ft09. The 4096 cap was NEVER a pool-truncation; it")
print("     was the INTENDED budget limit, which is the case a bigger max_tokens can address.")
print("  -> at K=4 (the eval-framework shape) the SAME pinned pool leaves ft09 only 3849 cells,")
print("     i.e. BELOW the shipped 4096 default: raising max_tokens there buys nothing until")
print("     n_ctx rises with it.")
