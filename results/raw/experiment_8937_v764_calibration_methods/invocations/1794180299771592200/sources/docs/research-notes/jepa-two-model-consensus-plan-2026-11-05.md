# JEPA and world-model planning: two-model review and consensus plan (2026-11-05)

Status: planning note. Nothing here was run. Gates are proposals and are not frozen until a task prompt freezes them.

## Method

Two independent researchers received the same written brief. They had no sight of each other's work.

| Researcher | How it ran | Web access |
|---|---|---|
| Fable 5.1 | Claude subagent | Yes (WebSearch, WebFetch) |
| GPT-6 Astra | `codex exec -m gpt-6-astra -s read-only` | Yes (its log shows web searches) |

Both were told: read only, no GPU, cite every external claim, label unverified claims. The outer loop then compared the two reports and spot-checked repo facts.

## Where both agree

1. **H-JEPA is moderate, narrow evidence.** Visual AntMaze 18.0% to 73.3% (3 seeds, simulation, continuous actions). Push-T fails at depth 3. DROID is offline path fidelity, not task success. The talk's "guardrails cannot be jailbroken" claim is an opinion with no measurement.
2. **No evidence yet for discrete or symbolic planning with a JEPA.** Fable found a probe-only paper (DWMR). Astra found a Crafter repo with 2 training seeds and a chess paper that dropped the JEPA loss.
3. **Do not do:** ARC work (suspended), any weight update to the Qwen3.8-27B generator, citing `experiment_2550_jepa_real_eval.json` (flagged), or crediting a win judged by the same oracle that defines correctness.
4. **Data and label provenance come first.** Any energy or calibration claim needs labels that the evaluated verifier did not produce.
5. **GPU work must be reserved through the conductor.** GPU 1 is detached. Immediate steps should be CPU only.

## Where they differ

| Topic | Fable | Astra | Outer-loop call |
|---|---|---|---|
| Test domain | Existing FoVer text traces | A new discrete substrate (MiniGrid DoorKey) | Both, in that order. FoVer data already exists. |
| LLM-JEPA step energy (2B LoRA) | Near-term step | Not proposed | Keep as conditional. Energy reranking hurt before. Needs a GPU slot. |
| Calibration pilot | Not proposed | Immediate and central | Adopt. It serves the calibrated-decision floor directly. |
| Hierarchy | Apply to prefix-then-step cost on FoVer | Only after a flat pilot passes | Gate hierarchy behind the monotonicity audit and the flat pilot. |
| AMI Labs funding | UNVERIFIED (press only, primary site silent) | "Verified" via an investor's page | Treat as unverified. The funding is not relevant to the plan. |
| Prior art | Found no external-energy-plus-JEPA planner | Found SAGE (arXiv 2603.02650, JEPA consistency energy ranks trajectories) | Read SAGE before any design. Not checked by the outer loop. |

## Repo facts the outer loop checked

- `results/experiment_4381_biprm_detector_localization_abstention.json` exists. Verdict: `complete: clean_powered_null_bidirectional_not_actionable`. First-error localization is 11 of 114 (0.0965). `verifier_is_oracle` is false.
- All three direction arms score exactly 0.0965 and the delta CI is [0.0, 0.0]. That pattern is suspicious: the localizer may be degenerate. A gate of "above 0.096" would be too weak against it.
- `data/fover_corpus.jsonl` has 8,829 rows. The v4 calibration corpus has 114 incorrect rows (Astra's log, Fable's n).
- 107 `results/` files match "jepa". Fable's audit named 20 JEPA experiment artifacts. The audit denominator must be fixed first.
- FoVer step order lives inside `step_text`. Any trajectory test needs the exp4381 trace reconstruction.

## Consensus plan

Anchor: the verifier-moat question (`GAP-ORACLE-DISTINCT`) and the calibrated-decision floor. The north-star file still names ARC as the destination, and ARC is suspended. The operator should confirm the interim anchor.

### Immediate (within 1 week, CPU only)

1. **Evidence readiness audit.** Trace labels for the FoVer and calibration corpora to independent sources (100 incorrect, 100 correct). Confirm that trace reconstruction from exp4381 works. Gate: no answer-derived features, no unresolved provenance. A negative result blocks every step below.
2. **Cost-monotonicity audit** (Fable's step 1, with two fixes). Per trace, test whether ensemble energy rises after the first labeled error. Add a degenerate-localizer check (why are all arms identical) and a planted-effect positive control. Proposed gate: rise in at least 60% of error traces, bootstrap CI95 lower bound above 50%, n = 114 error plus 114 matched correct. A negative result drops the landscape idea for the text domain.
3. **JEPA lineage naming audit** (Fable's step 2). Classify every JEPA-named artifact and module: predictor loss present, data source, n, seed, adversarial flag. Fix the denominator first. Tag any "JEPA" with no predictor-in-embedding loss as a naming-honesty candidate. This is hygiene and closes a known unproven record.

### Near-term (within 1 month)

4. **Calibration pilot** (Astra). Reuse `python/carnot/autoresearch/calibrated_decision_benchmark.py`. Needs at least 200 independently labeled errors. The v4 corpus has 114, so this step is blocked until step 1 finds or builds more labeled errors. Gate: Brier improvement over prevalence-only and logistic calibration, paired CI above 0, no worse accepted-error rate at 50% coverage. Serves the mandatory calibrated-decision floor.
5. **Flat discrete-JEPA verifier pilot** (Astra, MiniGrid DoorKey). A methods pilot, not a moat test: the simulator is an exact oracle, so a win says the learned model helps, not that Carnot has a moat. Needs an exact-model positive control, 5 seeds, matched latency, and a model under 1M parameters. Reserve a GPU slot through the conductor. Add SIGReg here only if training a latent.
6. **Hierarchy test, contingent.** Run the two-level cost (prefix energy then step energy) only if step 2 shows a landscape, and only after step 5's flat arm passes.
7. **LLM-JEPA step energy at 2B or smaller, conditional.** Only if steps 2 and 4 are positive. Prior energy-rerank attempts hurt. Contested between the two researchers.

### Parked

- Reproducing EB-JEPA Two Rooms. It is lowest priority. It adds an unreproduced base only if we ever build on it.
- The synthetic collapse-detector validation. It matters only when a latent-training task exists.

## What would change this plan

- If step 1 fails, stop all energy and calibration claims on this corpus. Build labeled data first.
- If step 2 is negative, drop the landscape and hierarchy ideas for text. The calibration and discrete-pilot steps still stand.
- If SAGE already shows a JEPA consistency energy that ranks trajectories, step 5 may be redundant. Read it first.

## Observation about the two researchers

Both reports were substantive, cited sources, labeled unverified claims, and proposed gates with sample sizes. Astra's report was shorter in steps (two immediate, two near-term) and noticed that its own calibration gate would be blocked by the 114-error corpus. No sign of covert under-delivery is visible in one sample. One sample proves nothing. This links to the covert-underperformance note of the same date.

## Rule checks

- Circularity: steps 4 to 7 must declare `verifier_is_oracle` and avoid labels the evaluated verifier produced.
- No weight update to the mandated generator.
- ARC stays suspended.
- Decentralization: all steps use open local models and open data.
