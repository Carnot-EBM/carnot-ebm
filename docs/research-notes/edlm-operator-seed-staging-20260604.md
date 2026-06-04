# EDLM Operator Seed Staging - 2026-06-04

**Status:** OPERATOR-GATED staging package. The loop does NOT seed EDLM: it clones nothing, trains nothing, and runs no model.

## One-Command Seed

```bash
git clone https://github.com/MinkaiXu/Energy-Diffusion-LLM.git && cd Energy-Diffusion-LLM && git checkout main && echo 'Seed ready'
```

This is the complete operator action needed to seed `.350`. Running it is the operator's call; this staging note does not run it.

## Tiny-Scale Kill-Gate Design

Part (a) stability mirrors Thesis-A `.341`: run only a tiny EDLM fit smoke under `.venv/bin/python` on the internal 3090, with a hard cuda-block before training. The first question is whether the tiny EDLM can train stably inside a bounded budget without Diverges/NaN events or obvious memory no-headroom.

Part (b) is the matched-COMPUTE comparison. Compare residual EDLM generation against an autoregressive baseline at equal total inference FLOPs. EDLM's iterative diffusion plus sequence-level energy correction gets no free pass for extra decoding passes. A win counts only at equal compute, with the same P0.1/Thesis-A trap explicitly closed.

Honest-negative exit: if tiny training diverges, produces NaN, cannot fit inside the bounded internal 3090 budget, or the AR/corpus setup has no headroom, EDLM is bounded at small scale for this route and STOP. Do not scale, do not reinterpret an invalid no-headroom comparison as a result.

## Decision Readiness

The operator can seed `.350` by running the one command above. Everything after that is the `.350 roadmap`: vendor+audit, tiny-EDLM fit smoke, matched-compute harness, and kill-gate verdict.

## Boundary

EDLM tests a different mechanism from both bounded routes: discrete diffusion with a sequence-level energy correction, not energy selection and not Thesis-A energy-as-sole-generator. This note does not claim EDLM sidesteps the bounds; it only stages the falsifiable operator seed.

## Provenance

- Exp 3793 preflight: `/home/ianblenke/github.com/ianblenke/carnot/results/experiment_3793_edlm_no_train_preflight_readiness.json`
- Exp 3781 feasibility scoping: `/home/ianblenke/github.com/ianblenke/carnot/results/experiment_3781_edlm_next_thesis_feasibility_scoping.json`
- Phase-3 menu framing: `/home/ianblenke/github.com/ianblenke/carnot/docs/research-notes/phase3-alternative-thesis-menu.md`
