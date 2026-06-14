# SOTA ingestion 2026-06-14: verifier-as-reward map for .390

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_as_reward_mapped_v390`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Spurious Rewards de-confounding control`, arxiv_id_or_url: `2506.10947`, url: `https://arxiv.org/abs/2506.10947`}
  - {name: `Memorization-shortcut diagnostic`, arxiv_id_or_url: `2601.11061`, url: `https://arxiv.org/abs/2601.11061`}
  - {name: `Youden-J reward-noise gate`, arxiv_id_or_url: `2601.04411`, url: `https://arxiv.org/abs/2601.04411`}
  - {name: `RLEF execution-feedback baseline`, arxiv_id_or_url: `2410.02089`, url: `https://arxiv.org/abs/2410.02089`}
  - {name: `Aletheia code-verifier recipe`, arxiv_id_or_url: `2601.12186`, url: `https://arxiv.org/abs/2601.12186`}
  - {name: `CodeScaler reward-model baseline`, arxiv_id_or_url: `2602.17684`, url: `https://arxiv.org/abs/2602.17684`}
  - {name: `Self-distilled process-reward fork`, arxiv_id_or_url: `2604.03128`, url: `https://arxiv.org/abs/2604.03128`}
  - {name: `Budget-aware verifier plus self-consistency hybrid`, arxiv_id_or_url: `2510.14913`, url: `https://arxiv.org/abs/2510.14913`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v390: `non_qwen_same_generator_random_label_ablation_v390`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner (e.g. the non-Qwen replication or the math-process-reward fork).

## Fresh-pass provenance

Read `research-references.md` `.389 planning sweep`,
`research-studying.md`, and
`results/experiment_4199_verifier_reward_decisive_a_vs_b_collect.json`.
Exp 4199 is not a completed A-vs-B result: it reports
`blocked_gate_check_failed` because the upstream
`exp4198-verifier-reward-3arm-rft-launch.training_launched` gate was false.
This note therefore maps the literature to .390 planning and does not promote
the blocked A-vs-B collection as evidence.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "spurious rewards RLVR random rewards verifiable rewards Youden J" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "RLVR code execution feedback verifier reward random feedback Aletheia" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "self distilled RLVR contrastive evidence process reward model ThinkPRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "budget aware test time scaling discriminative verification solve verify" --limit 8`

The cluster helper emitted broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for all four focused queries, so no
S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch verified
arXiv:2506.10947, arXiv:2601.11061, arXiv:2601.04411, arXiv:2509.20837,
arXiv:2410.02089, arXiv:2601.12186, arXiv:2602.17684, arXiv:2604.03128,
arXiv:2605.19436, arXiv:2510.14913, arXiv:2504.01005, and arXiv:2504.16828.

## Exp 4199 A-vs-B status

Exp 4199 is `blocked_gate_check_failed`, with the failed gate
`training_launched == true` actually observed as false. The literature mapping
therefore treats the de-confounded A-vs-B as still open. The .390 experiment
must run the non-Qwen base and same-generator random-label ablation before
claiming that verifier labels carry training signal beyond distillation,
generator prior, or spurious reward structure.

## SOTA -> experiment mapping

## CONFOUND hardening

**Method/source:** Spurious Rewards: Rethinking Training Signals in RLVR,
arXiv:2506.10947 (https://arxiv.org/abs/2506.10947), shows that random rewards
can recover much of a Qwen2.5-Math RLVR gain, while the effect is model-family
dependent. Spurious Rewards Paradox, arXiv:2601.11061
(https://arxiv.org/abs/2601.11061), gives the memorization-shortcut diagnostic.
RLV-epsilon-R, arXiv:2601.04411 (https://arxiv.org/abs/2601.04411), gives the
Youden-J reward-noise gate. Verification Limits Code LLM Training,
arXiv:2509.20837 (https://arxiv.org/abs/2509.20837), keeps verification
calibrated rather than over-rigid.

**Carnot stack mapping:** Use the execution verifier as reward only inside a
non-Qwen base replication with a same-generator random-label arm, Youden-J
reporting, and a memorization-shortcut diagnostic.

**Implication:** The de-confounded A-vs-B is the literature's open question.
A true Carnot positive is Arm A beating Arm B on held-out code while the
verifier has J > 0 and shortcut diagnostics stay clean.

**Failure mode:** A Qwen-only lift, a verifier-only lift without the random
label arm, or a lift with shortcut activation remains compatible with the
spurious-reward confound.

**Experiment mapping:** Flag the .390 non-Qwen same-generator random-label
replication. Report pass@1 delta, bootstrap CI, TPR, FPR, Youden-J, truncation,
and shortcut diagnostics for Arm A and Arm B.

## Code-RLVR baselines

**Method/source:** RLEF, arXiv:2410.02089
(https://arxiv.org/abs/2410.02089), is the execution-feedback RL baseline.
Aletheia, arXiv:2601.12186 (https://arxiv.org/abs/2601.12186), supplies the
code-verifier RLVR recipe. CodeScaler, arXiv:2602.17684
(https://arxiv.org/abs/2602.17684), sets an execution-free code reward-model
frontier. Scaling Agentic Verifier, arXiv:2602.04254
(https://arxiv.org/abs/2602.04254), is an additional test-time code verifier
frontier from the .389 sweep.

**Carnot stack mapping:** These define the baseline table a positive .390
result must beat or distinguish itself from: execution feedback RL, trained
code verifier recipe, reward-model scaling, and active discriminative test
generation.

**Implication:** A verifier-as-reward claim is not just "RFT helped"; it must
show the verifier label adds signal over established code-RLVR and reward-model
baselines under matched budgets.

**Failure mode:** These baselines can improve code accuracy while still
leaving the label-causality confound unresolved.

**Experiment mapping:** Add a .390 comparison table with RLEF-style execution
feedback, Aletheia verifier recipe settings, CodeScaler reward-model rows, and
the Carnot A-vs-B arms.

## Process/self-distill fork

**Method/source:** Self-Distilled RLVR, arXiv:2604.03128
(https://arxiv.org/abs/2604.03128), keeps verifier reward as update direction
while using self-distillation for token-level magnitude. CEPO, arXiv:2605.19436
(https://arxiv.org/abs/2605.19436), sharpens dense credit with contrastive
positive and negative evidence. ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), is the expensive generative
process-verifier comparator.

**Carnot stack mapping:** This is the .390 math-process-reward fork after the
de-confounding gate: sparse verifier direction first, dense process credit
second.

**Implication:** If A beats B cleanly, Carnot can test whether dense
process-reward modulation improves sample efficiency without losing verifier
grounding.

**Failure mode:** Privileged self-distillation can leak answers and process
verifiers can be expensive or locally valid but globally wrong. This fork
should not replace the non-Qwen A-vs-B gate.

**Experiment mapping:** Queue sparse verifier reward versus RLSD modulation
versus CEPO-style contrastive evidence on math traces, with ThinkPRM as the
quality ceiling and cost comparator.

## Cost-crossover hybrid

**Method/source:** Budget-aware Test-time Scaling via Discriminative
Verification, arXiv:2510.14913 (https://arxiv.org/abs/2510.14913), supports
cheap discriminative verifier plus self-consistency as the practical hybrid.
When To Solve, When To Verify, arXiv:2504.01005
(https://arxiv.org/abs/2504.01005), sets the fixed-budget solve-versus-verify
bar.

**Carnot stack mapping:** Report the verifier as a cost-normalized hybrid with
self-consistency, not as a raw accuracy-only judge replacement.

**Implication:** A .390 result should include verifier-only, self-consistency,
and verifier-plus-self-consistency rows with matched compute.

**Failure mode:** A hybrid can be an inference-time efficiency win while still
not proving verifier labels are causally useful for training.

**Experiment mapping:** Carry fixed candidate budget, verifier-call budget,
latency, token cost, and cost per accepted correct solution into the .390
table.

## Flagged for .390

`non_qwen_same_generator_random_label_ablation_v390` is the strongest next
method. The reason is not that process rewards are unimportant; it is that
arXiv:2506.10947 and arXiv:2601.11061 make the A-vs-B label-causality
question load-bearing. The math-process-reward fork should follow only after
the non-Qwen same-generator random-label control has been run and interpreted.

