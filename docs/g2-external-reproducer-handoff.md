# G2 External-Reproducer Handoff — FoVer Headline AUROC

**Audience:** anyone who is **not** the operator (Ian Blenke) and wants to close
publication gate **G2** (independent reproduction) — the *sole* remaining blocker
to `paper_ready` (`ops/north-star.md` §2). This page is the turnkey package: one
clone, one install, one command.

This is **cheap and CPU-only**. The headline is the verifier ensemble *scoring a
labeled corpus* — no GPU, no 35B model, no HuggingFace credentials.

---

## One-command reproduction

```bash
git clone https://github.com/Carnot-EBM/carnot-ebm && cd carnot-ebm
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
JAX_PLATFORMS=cpu python3 scripts/reproduce_fover_headline.py
```

That script **exits non-zero unless** both numbers land inside their published
confidence intervals, so a zero exit (`echo $?` -> `0`) *is* the pass.

## The two assertions (what a green run proves)

| Quantity | Must land in | Published value |
|---|---|---|
| condition-A (production) mean AUROC | `[0.9027, 0.9235]` | 0.9131 |
| learning_contribution (FR-11 ablation) mean | `[0.0125, 0.0245]` | 0.0185 |

Both over **n=1,000**, **5 seeds** `[42, 137, 271, 314, 1729]`.

## Corpus checksum (confirm you cloned the measured corpus)

```
sha256(data/fover_corpus.jsonl) = 585a4b8099fae140b850e10d36f121e0a25e645c8d0d264936d5efa0a62b330f
```

```bash
sha256sum data/fover_corpus.jsonl   # compare against the value above
```

The corpus is committed (no separate download). It is Carnot's derivation of the
public FoVer step-error dataset, traceable to source.

## The zero-effort path: the GitHub Actions workflow

A non-operator with write access to a fork can close G2 *without a local clone*:
open the **Actions** tab -> **"FoVer Headline Independent Reproducer"** ->
**"Run workflow"**. It runs on a clean `ubuntu-latest` runner
(`.github/workflows/reproduce-fover-headline.yml`):
`checkout` -> Python 3.12 -> `pip install -e .` ->
`python3 scripts/reproduce_fover_headline.py`. A green run on GitHub-hosted
infrastructure is non-operator evidence. It also runs weekly (Mon 07:00 UTC).

This workflow has been **dry-run green** in an isolated clean-room container
(`python:3.12-slim`, fresh `pip install -e .`): the assert command exited `0`
with condition-A AUROC `0.9131` and learning_contribution `0.0185` — both in
CI. See `results/experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.json`.

## Exactly what closes G2

G2 requires **>=1 reproducer who is NOT the operator**. The Phase-1 ship gate
counts **a CI run** as that reproducer. So G2 closes when **either**:

1. The GitHub Actions workflow above runs green on GitHub infrastructure
   (triggered by anyone other than the operator), **or**
2. A non-operator runs the one-command reproduction on their own machine and
   reports condition-A AUROC in `[0.9027, 0.9235]` and learning_contribution in
   `[0.0125, 0.0245]`.

Then record it per `ops/reproduction-runbook-fover-headline.md` ("How to record
a successful reproduction") — set `g2_independent_reproducer: true` in
`ops/publication_gate_state.json` with the evidence, and
`python3 scripts/publication_gate.py` will report `paper_ready=True`.

## What this handoff does NOT claim

It does **not** claim G2 is met. Autonomous work can build and *dry-run* the
mechanism (which this package proves runs green), but only an actual external/CI
run by a non-operator flips `g2_independent_reproducer` to true.

## Cross-references

- `ops/reproduction-runbook-fover-headline.md` — full protocol + caveats
- `.github/workflows/reproduce-fover-headline.yml` — the CI mechanism
- `scripts/reproduce_fover_headline.py` — the harness (the assert lives in `main()`)
- `ops/north-star.md` §2 — the G1–G4 gate definition
