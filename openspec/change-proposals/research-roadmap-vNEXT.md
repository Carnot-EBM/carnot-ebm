# Research Roadmap vNEXT - 2026.06.465 - Repair Phase D execution, audit the positive signal, and replace raw replay with audited self-learning

**Milestone:** 2026.06.465
**Planner:** Codex GPT-5, 2026-06-30 (UTC)
**Prior milestone:** 2026.06.464 (PHASE D power + confirmation)
**Theme:** `.464` completed all tasks, but the capstone correctly stayed `execution_incomplete`: the
best verifier evidence remained a blocked D1 artifact with a promising +8pp MuSR signal, D6 never ran
because the SOTA/judge preflight failed, D4 produced a strong second-corpus confirmation that still needs
an independent no-leak audit, and FR-11 replay memory hurt held-out accuracy. `.465` therefore does not
add a new moat family. It repairs the execution path, reuses the mandated SOTA local GGUF cache, audits
the two positive signals, reruns D6 as a tool-first cascade that can execute without brittle confidence
telemetry, and replaces raw replay memory with audited skill-graph self-learning.

---

## 1. What .464 proved

| Area | .464 result | Read for .465 |
|---|---|---|
| SOTA/Judge preflight | `blocked_judge_server`; all three mandated GGUF repos resolved locally, but judge/confidence path was not ready | Split readiness into `sota_models_ready`, `sota_judge_ready`, and `tool_first_verifier_ready`. D6 must not depend on one brittle boolean. |
| D1 powered LoRA-EBM/EORM | `blocked_sota_candidate_refresh_unavailable`, but powered scorer was available and reported 0.665 vs tuned-SC 0.585 on MuSR (`delta=+0.080`, CI touches zero, n=200) | Audit and refresh candidates before any headline claim. If refreshed D1 clean-null executes properly, retire this shape. |
| D2 process reward repair | `complete_process_reward_no_win_musr_minus_0p030` | Do not spend another slot on scalar/process-reward repair unless it is part of a differentiated cascade or audit. |
| D3 KAN/PURM | `complete_kan_purm_no_incremental_lift_over_powered_d1_minus_0p060` | KAN/PURM calibration did not improve the powered D1 arm. Treat as negative unless a later audit finds a data/field issue. |
| D4 second corpus | `success_second_corpus_confirms_musr_margin_constraintbench_exact_v1_plus_0p370` | Promising but not enough by itself because D1/D6 execution was incomplete. Audit leak/oracle controls and re-evaluate the best executable arm. |
| D6 cascade | gated/blocked because `exp5043.sota_judge_ready=false`; no artifact | Rerun as tool-first T1/SAFE cascade with optional SOTA judge fallback, not as a judge-server-only task. |
| Moat gate | `complete_moat_execution_incomplete_v464_blocked_or_missing_phase_d` | Gate logic worked: it refused to headline positive but incomplete evidence. `.465` must provide complete artifacts or bounded retirement. |
| FR-11 self-learning | `complete_verifier_trace_self_learning_replay_memory_minus_0p050`; held-out accuracy fell from 0.70 to 0.65 | Raw replay memory is not credible positive self-learning. Use audited skill graphs, self-audit, nonforgetting guards, and no-promote rules. |
| KV260 | `success_kv260_pbit_timing_ratio_packet_built` | Board path is live. Next hardware slot should add reproducible board transcript/testbench evidence, not broader speedup claims. |
| SOTA ingestion | `success_sota_ingestion_v465_actionable_references_added` | `.465` has enough current literature hooks; it should execute them before expanding the backlog. |
| ARC | `complete_tu93_no_new_level_residual_duplicate_depth`; no new bank | ARC remains opportunistic. Do not propose duplicate level solves; improve live self-discovery only. |

## 2. The three biggest gaps to the PRD vision

1. **Execution gap:** the PRD requires verifiable, reproducible constraint-reasoning wins. The strongest
   current verifier evidence is still behind blocked candidate-refresh and judge/cascade prerequisites.
2. **Generalization/audit gap:** the D4 second-corpus result is large, but it is not yet independently
   audited against leakage, oracle use, duplicate rows, and dependence on a blocked D1 upstream.
3. **Continuous learning gap:** FR-11 requires autonomous directed self-learning. `.464` ran a loop, but
   the loop reduced held-out performance, so `.465` must add audited memory promotion and nonforgetting
   gates rather than simply inserting more replay traces.

## 3. Fresh research incorporated before experiment design

The `.465` planner added a synthesis section at the top of `research-references.md`, grounded in the
Exp5053 `.465` ingestion set plus a live web sweep. The most actionable research hooks are:

- **T1 / SAFE tool-first verification** (`arXiv:2504.04718`, `arXiv:2604.01993`): use deterministic
  tools and evidence-grounded checks before expensive judge fallback. This directly addresses the D6
  blocked-judge failure.
- **In-Writing / delayed constrained decoding** (`arXiv:2601.07525`): rebuild SOTA candidate refresh
  through structured constraint fields when top-logprob telemetry is unavailable.
- **Vegas and Reward-Guided Decoding** (`arXiv:2602.07223`, `arXiv:2605.28020`): add a cost-frontier arm
  that measures accuracy per generated token, verifier call, and judge call.
- **SAVeR / Audited Skill-Graph Self-Improvement / constraint-guided reasoning** (`arXiv:2604.08401`,
  `arXiv:2512.23760`, `arXiv:2606.26108`): convert FR-11 from raw replay memory into audited skill
  promotion with no-promote and nonforgetting guards.
- **Structured Testbench Generation and p-bit hardware references** (`arXiv:2606.12983`, prior p-bit
  FPGA work): keep hardware evidence board-local, transcript-backed, and parity/timing focused.

Extropic TSU and Logical Intelligence Kona remain strategic architecture signals only. No `.465`
experiment may claim external hardware parity or speedup from those sources.

## 4. Architecture and dependency graph

```
                                 exp5056 PHASE 0
                 archive .464 close-state and activate .465 records
                                         |
                                         v
                         exp5057 gate-state preflight
        split SOTA model cache, judge/confidence, tool-first verifier readiness
                                         |
                   +---------------------+----------------------+
                   |                                            |
                   v                                            v
       exp5058 SOTA candidate refresh              exp5065 KV260/testbench continuity
       delayed constrained decoding                board transcript, no speedup claim
                   |
                   v
       exp5059 D1 refreshed-audit scorer
       MuSR powered signal, frozen-old vs refreshed candidates
                   |
       +-----------+------------------+-----------------+
       |                              |                 |
       v                              v                 v
 exp5060 D4 second-corpus audit  exp5061 D6 tool-first  exp5062 guided decoding
 no-leak / no-oracle controls    cascade with fallback   cost frontier
       |                              |                 |
       +------------------------------+-----------------+
                                      |
                                      v
                    exp5063 moat gate resolution v465
          realized / MuSR-scoped / bounded-retired / execution-incomplete

Parallel continuous/reserved slots:

    exp5064 FR-11 audited skill-graph self-learning
    exp5066 SOTA ingestion for .466
    exp5067 ARC live-path self-discovery, no duplicate solves
                                      |
                                      v
                              exp5068 capstone
```

Structured gate edges in `research-roadmap-next.yaml`:

- `exp5058` runs only if `exp5057.sota_models_ready == true`.
- `exp5059` runs only if `exp5058.candidate_refresh_ready == true`.
- `exp5060` runs only if `exp5059.best_arm_available == true`.
- `exp5061` runs only if `exp5057.tool_first_verifier_ready == true` and `exp5059.best_arm_available == true`.
- `exp5062` runs only if `exp5058.candidate_refresh_ready == true` and `exp5059.best_arm_available == true`.

## 5. Phases

### Phase 0 - Transition and state capture

- **exp5056:** archive the `.464` close-state and activate `.465` records. This task must record that the
  prior milestone completed all tasks but ended execution-incomplete because D1 candidate refresh and D6
  cascade were blocked, despite D4 and KV260 successes.

### Phase A - Execution repair

- **exp5057:** write a gate-state summary and runtime preflight. It must distinguish local SOTA model
  readiness from judge readiness and from tool-first verifier readiness. It should also leave a short
  machine-readable skip reason for downstream tasks.
- **exp5058:** rebuild the SOTA candidate refresh path using delayed constrained decoding and structured
  candidate rows. This is the direct repair for the D1 `blocked_sota_candidate_refresh_unavailable` state.

### Phase D - Verifier moat execution

- **exp5059:** audit and rerun the D1 powered signal on refreshed candidates, while also freezing the old
  candidate cache for an apples-to-apples comparison. The deliverable must decide whether the +8pp signal
  survives a real refresh.
- **exp5060:** independently audit the D4 second-corpus confirmation. Required controls: row hashes,
  no-oracle candidate provenance, duplicate/leak checks, and a comparison against genuine tuned-SC.
- **exp5061:** rerun D6 as a tool-first cascade: deterministic checks, SAFE-style evidence checks, cheap
  verifier, and optional SOTA local GGUF judge fallback. If judge telemetry is unavailable, the task should
  still produce a bounded tool-first artifact.
- **exp5062:** measure guided decoding and verifier-cost frontier. The intervention must create genuinely
  different candidates and report generated tokens, verifier calls, judge calls, latency, and accuracy.
- **exp5063:** resolve the moat gate for `.465`: realized, MuSR-scoped positive, bounded retirement, or
  execution-incomplete. The gate must not headline blocked artifacts or un-audited second-corpus wins.

### Phase E - Continuous self-learning

- **exp5064:** replace raw replay memory with audited skill-graph self-learning. The loop must mine
  near-misses, create self-audited/verifier-audited skills or memory entries, evaluate held-out performance,
  and refuse promotion when the held-out delta is non-positive.

### Phase C/R - Hardware, SOTA, and ARC continuity

- **exp5065:** extend the KV260 p-bit/timing-ratio packet with board transcript and structured testbench
  evidence. Optional GateMate/PolarFire checks are allowed only as prechecks, not as speedup claims.
- **exp5066:** reserved SOTA ingestion for `.466`, focused on sources that directly change the next
  experiment plan and de-duplicated against Exp5053.
- **exp5067:** opportunistic ARC live-path self-discovery. It must run registry precheck first, avoid
  duplicate solves, and include `solve_provenance: live_agent_self_discovery` for any solve claim.
- **exp5068:** capstone: reconcile artifacts, state the `.465` moat and FR-11 verdicts, and choose the
  `.466` route.

## 6. Falsifiable gates

- **Moat realized:** a properly executed, oracle-distinct verifier or cascade beats genuine tuned-SC on
  MuSR with CI excluding zero and either passes the D4 audit or reaches a D6 cost/accuracy efficiency win.
- **MuSR-scoped positive:** refreshed D1 remains positive on MuSR, but D4 audit or D6 cascade is blocked,
  negative, or inconclusive. This is progress, not a PRD-level moat claim.
- **Second-corpus scoped positive:** D4 remains positive after leak/no-oracle audit, but MuSR D1/D6 do not
  execute cleanly. This is a transfer clue, not a headline verifier-moat claim.
- **Bounded retirement:** refreshed D1, D4 audit, and tool-first D6 all execute and clean-null or regress
  on headroom-present data. If the same blocked or negative verdict recurs for a prior failed scope, the
  task's `prior_failures.retire_if_same_verdict: true` entry makes retirement mechanical.
- **Execution incomplete:** any missing required artifact field, blocked SOTA model path, unbuilt candidate
  cache, skipped D6 without tool-first artifact, skeleton training, degenerate abstention, or missing paired
  statistics.
- **FR-11 positive evidence:** held-out delta is positive, contamination guard passes, promoted skills are
  self-audited and externally verifier-audited, and no nonforgetting guard fails.
- **FR-11 guarded negative:** loop executes but no-promote triggers or held-out delta is non-positive. This
  is an honest negative, not a failure to run.

## 7. Hardware requirements

- **Dual RTX 3090 CUDA local runtime:** preferred for mandated SOTA GGUF inference, candidate refresh,
  judge fallback, LoRA/EORM scoring, and guided decoding. Do not iGPU-pin Phase D headline runs.
- **Mandated SOTA local GGUF models:** every LLM-dependent experiment must include at least one of
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF` in `MODEL_SPECS`. Legacy small models are smoke tests only.
- **KV260 over SSH (`ssh kria`):** allowed for board-local overlay/timing/testbench checks. No host SD-card
  operations. No speedup claim without board transcript and CPU reference parity.
- **GateMate and PolarFire:** optional prechecks only in `.465`; no Carnot latency/speedup claim unless a
  flashed or dispatched board transcript is included.
- **Extropic TSU / external thermodynamic hardware:** architecture context only. No local experiment may
  imply access or performance equivalence.

## 8. Expected deliverables

| Experiment | Primary deliverable |
|---|---|
| exp5056 | `results/experiment_5056_archive_464_activate_465.json` |
| exp5057 | `results/experiment_5057_gate_state_preflight_v465.json` |
| exp5058 | `results/experiment_5058_sota_candidate_refresh_inwriting.json` |
| exp5059 | `results/experiment_5059_d1_sota_refresh_audit.json` |
| exp5060 | `results/experiment_5060_second_corpus_audit_v2.json` |
| exp5061 | `results/experiment_5061_tool_first_cascade.json` |
| exp5062 | `results/experiment_5062_guided_decoding_cost_frontier.json` |
| exp5063 | `results/experiment_5063_moat_gate_resolution_v465.json` |
| exp5064 | `results/experiment_5064_audited_skillgraph_self_learning.json` |
| exp5065 | `results/experiment_5065_kv260_testbench_timing_packet.json` |
| exp5066 | `results/experiment_5066_sota_ingestion_v466.json` |
| exp5067 | `results/experiment_5067_arc_live_path_self_discovery.json` |
| exp5068 | `results/experiment_5068_capstone_v465.json` |

## 9. Why this is the natural next milestone

`.464` already answered the broad-planning question: the positive D1/D4 evidence is worth repairing and
auditing, while D2/D3 and raw replay memory should not be repeated as-is. `.465` is therefore an execution
milestone. It makes the blocked gates cheap and explicit, moves D6 to a tool-first architecture that can
produce evidence without a fragile judge endpoint, audits the second-corpus result before using it as
confirmation, and replaces negative replay-memory self-learning with an audited no-promote loop aligned
with FR-11. If `.465` executes cleanly and still cannot produce a robust win, bounded retirement becomes
credible; if it succeeds, the program finally has a properly audited verifier-moat and self-learning path
to scale in `.466`.
