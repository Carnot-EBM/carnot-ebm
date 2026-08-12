"""Spec: REQ-ARC-WMTE-6227.

Regression test for the stale worst-case-prompt-token-count bug.

docs/research-notes/live-agent-adversarial-review-2026-08-08.md, "Correctness" section,
major finding 5:

  "The shared-context-pool admission arithmetic is sized against a stale worst-case prompt
  measurement. `_INDUCE_WORST_CASE_PROMPT_TOKENS = 15767` was measured 2026-07-28 at k=8 with
  no object table; since then k defaults to ALL transitions (2026-08-01) and the object-
  structure block is default ON (2026-08-07, up to two 60-row tables), while the pool leaves
  only ~617 tokens of per-slot margin ... Fix: re-measure the worst case under current
  defaults through the tokenizer; cap the objects/transitions blocks by token budget, not row
  count."

THE FIX. `_INDUCE_WORST_CASE_PROMPT_TOKENS` moved 15767 -> 22352, re-measured through the real
gemma-4-31B-it Q4_K_M GGUF tokenizer (`llama_cpp.Llama(vocab_only=True)`) against ONE worst-case
`induce_prompt()` call (64x64 grid, 25 transitions, same rng seed 5900 as the 2026-07-28
predecessor measurement) under CURRENT defaults: `k=None` (all transitions) and
`CARNOT_ARC_OBJECT_PERCEPTION` unset (its own default, object table ON).
`_default_induce_n_ctx()`'s arithmetic is unchanged and picks the new figure up automatically
(no code there needed to change) -- see `tests/python/test_arc_generator_vram_guard.py` for the
tests that pin the derived n_ctx and the VRAM-guard consequences of this change.

This test pins the constant's VALUE directly, rather than only through the derived n_ctx, so a
silent edit to the constant (independent of a real re-measurement) is caught here even if
someone forgets to re-derive the VRAM guard tests too.
"""

from __future__ import annotations

from carnot.agentic import arc_executable_world_model as wm


def test_worst_case_prompt_tokens_is_the_2026_08_08_remeasurement() -> None:
    assert wm._INDUCE_WORST_CASE_PROMPT_TOKENS == 22352, (
        "this constant must only change via a fresh re-measurement through the real tokenizer "
        "under CURRENT defaults (k=all transitions, object table on) -- see the constant's own "
        "comment for the methodology. If you just measured a new worst case, update this pin to "
        "match; do not silently edit the constant without updating this test."
    )


def test_default_n_ctx_derives_correctly_from_the_new_worst_case() -> None:
    """The admission arithmetic itself is unchanged code -- this pins that it still derives the
    right pool size from the new constant, so a future refactor of `_default_induce_n_ctx()`
    cannot silently decouple the two."""
    # Read the RESOLVER, not the raw constant (2026-08-11, REQ-ARC-WMTE-6253). K became
    # env-overridable via CARNOT_ARC_LLAMA_SERVER_SLOTS. Against the constant this test
    # asserted 212992 == 106496 and FAILED the moment an operator used the documented
    # knob -- a test that punishes the feature it is meant to protect. The env is cleared
    # here so the assertion pins the SHIPPED default regardless of the caller's shell.
    monkeypatch = None  # noqa: F841  (documented below: os-level clear, no fixture needed)
    import os

    prior = os.environ.pop("CARNOT_ARC_LLAMA_SERVER_SLOTS", None)
    prior_ctx = os.environ.pop("CARNOT_ARC_INDUCE_N_CTX", None)
    try:
        max_tokens = wm._INDUCE_DEFAULT_MAX_TOKENS
        slots = wm._llama_server_slots()
        need = slots * (wm._INDUCE_WORST_CASE_PROMPT_TOKENS + max_tokens)
        expected_n_ctx = ((need + 4095) // 4096) * 4096
        assert expected_n_ctx == 106496
        assert wm._default_induce_n_ctx() == expected_n_ctx
    finally:
        if prior is not None:
            os.environ["CARNOT_ARC_LLAMA_SERVER_SLOTS"] = prior
        if prior_ctx is not None:
            os.environ["CARNOT_ARC_INDUCE_N_CTX"] = prior_ctx


def test_new_worst_case_exceeds_the_old_per_slot_budget() -> None:
    """The load-bearing finding, asserted directly: the OLD pool (sized for the stale 15767
    figure) genuinely could not have held the new, real worst-case prompt at K=4 concurrency --
    this is not a cosmetic constant bump, it is a real overflow-risk fix."""
    old_n_ctx = 81920
    max_tokens = wm._INDUCE_DEFAULT_MAX_TOKENS
    slots = wm._LLAMA_SERVER_DEFAULT_SLOTS
    old_per_slot_budget = old_n_ctx // slots - max_tokens
    assert wm._INDUCE_WORST_CASE_PROMPT_TOKENS > old_per_slot_budget, (
        "the whole point of this fix: the real worst-case prompt must exceed what the STALE "
        "pool could admit per slot, or there was no overflow risk to fix in the first place"
    )
