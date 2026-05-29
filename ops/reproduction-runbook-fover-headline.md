# G2 Reproduction Runbook — FoVer Headline AUROC

**Purpose:** Close gate **G2** (independent reproduction) of the publication
gate (`ops/north-star.md` §2, `scripts/publication_gate.py`). G2 is the *sole*
remaining blocker to `paper_ready` (G1/G3/G4 are met). This runbook lets a
person who is **not the operator** re-run the headline experiment from a fresh
clone and confirm the number independently.

**Good news for the reproducer:** this is **cheap and CPU-only**. The headline
is the verifier ensemble *scoring a labeled corpus* — it does **not** load a
35B model or require a GPU (`live_model_invoked: False`, wall-clock ~16s).

---

## The exact claim being reproduced

> Carnot's verifier ensemble reaches **mean AUROC 0.9131** on the FoVer
> step-error corpus (**n=1,000**, **5 seeds**), under a dual-condition protocol:
> condition A (production, with the FR-11 session-memory verifier) = **0.9131**
> [CI95 0.9027–0.9235]; condition B (architecture-only, FR-11 memory removed)
> = **0.8947**; **learning contribution = +0.0185** [CI95 0.0125–0.0245].

Precise scope (do not overstate — these were checked against the artifact):
- **4 verifiers** contribute to the FoVer score: `fr11_session_memory`,
  `tier0r_curry_howard`, `tier0s_arithmetic_gap`, `tier0u_logical_consistency`.
  (The broader project ensemble is larger; only these score FoVer.)
- **Verifier-scoring against the labeled corpus**, not live LLM generation.
  `live_model_invoked: False`. CPU is sufficient.
- Source artifacts: `results/experiment_2837_fover_memory_leakage_v3.json`
  (primary, checksum `47872d20…`) and
  `results/experiment_2850_fover_dual_condition_integrity_v4.json` (v4 re-run).
- Fixed seeds: **[42, 137, 271, 314, 1729]**.

---

## Prerequisites (what the reproducer needs)

- A fresh `git clone` of `github.com/Carnot-EBM/carnot-ebm` — **the FoVer
  corpus is committed** (`data/fover_corpus.jsonl`, 8,829 labeled rows;
  `data/fover_corpus_v4.json`), so no separate dataset download is required.
  (The corpus is Carnot's derivation of the public FoVer dataset and is
  traceable to it.)
- Python 3.11+.
- `pip install -e .` (no `[cuda]` extra needed — this run is CPU-only).
- **No GPU, no 35B model, no HuggingFace credentials.** This is the key
  difference from the live-inference experiments.
- Disk: trivial (corpus is ~4 MB).

## Reproduction steps

```bash
git clone https://github.com/Carnot-EBM/carnot-ebm && cd carnot-ebm
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# Run the headline experiment with the published seeds:
python3 - <<'PY'
from carnot.eval.fover_memory_leakage_v3 import ExperimentConfig, run_experiment
cfg = ExperimentConfig(
    random_seeds=[42, 137, 271, 314, 1729],
    n_examples=1000,
)
result = run_experiment(cfg)
print("condition A (production) mean AUROC:",
      result["condition_a_production_auroc_mean"])
print("condition B (architecture-only) mean AUROC:",
      result["condition_b_architecture_only_auroc_mean"])
print("learning contribution:", result["learning_contribution_ci95"]["mean"])
print("reproducibility_checksum:", result.get("reproducibility_checksum"))
PY
```

(If `ExperimentConfig` needs `repo_root`/`results_dir`, pass them explicitly;
inspect `inspect.signature(ExperimentConfig)`.)

## Acceptance criteria (what counts as a PASS)

A reproduction **passes G2** if, from an independent clone on a non-operator
machine:

1. **condition-A mean AUROC ∈ [0.9027, 0.9235]** (the published CI95), AND
2. **learning_contribution mean ∈ [0.0125, 0.0245]** (the FR-11 ablation CI), AND
3. the run completes without falling back to a degraded/blocked path
   (`live_model_invoked` may be False — that is expected and correct here).

A byte-identical `reproducibility_checksum` match is a **bonus** (strongest
evidence) but is not required — checksum can legitimately differ across
platforms/library versions while the AUROC still lands in-CI. The AUROC-in-CI
result is the load-bearing acceptance.

## What makes the reproduction "independent" (so it actually closes G2)

- Run by **someone other than the operator** (Ian Blenke).
- From a **fresh clone**, on a **different machine**, with **no access to the
  operator's `results/` directory** (so they are recomputing, not reading).
- Report the exact numbers, platform, Python/lib versions, and seeds used.

## How to record a successful reproduction

When a reproduction passes, set G2 in `ops/publication_gate_state.json`:

```json
{
  "g2_independent_reproducer": true,
  "g2_evidence": "<who> reproduced FoVer headline on <platform> <date>: condition_A_auroc=<x> (in CI), learning_contribution=<y>; see <link/notes>",
  "last_reviewed": "<date>"
}
```

Then `python3 scripts/publication_gate.py` should report `paper_ready=True`
with `unmet_gates: none` — the gate closes and the paper is publishable on the
FoVer headline.

## Known caveats (be honest with the reproducer)

- **Upstream preflight dependency:** `ExperimentConfig` references an
  `exp2836_path` (a SOTA-runtime preflight artifact). For the CPU verifier-
  scoring path this is a gate-check, not a compute dependency, but if the run
  errors on a missing `results/experiment_2836_*.json`, that artifact (or a
  stub satisfying the precondition) must be present. Confirm before sending
  to an external reproducer.
- **`tier0s_arithmetic_gap`** is a class within the verify package (referenced
  in `python/carnot/verify/__init__.py`), not a standalone `tier0s_*.py`
  module — it is available via the package import, just not as its own file.
- **Corpus provenance:** `data/fover_corpus.jsonl` is Carnot's derived FoVer
  corpus. A maximally-independent reproducer could regenerate it from the
  public FoVer dataset, but using the committed file is acceptable for G2
  (it is the same corpus the headline was measured on, traceable to source).

## Recommended outreach

The Phase-1 external reproducer "CG" (`ops/external-reproducer-2026-05-26-cg.md`)
already ran `pip install carnot-ebm` + the tutorial. CG is the natural first
ask for G2 — this run is even cheaper (no GPU). A single CG reproduction within
CI closes G2 and flips `paper_ready` to true.

## Cross-references
- `ops/north-star.md` §2 (the gate), §1 (the headline claim)
- `scripts/publication_gate.py` (G1–G4 computation)
- `ops/publication_gate_state.json` (where G2 is recorded)
- exp2837 / exp2850 (source artifacts)
- `carnot.eval.fover_memory_leakage_v3` (the importable experiment)
