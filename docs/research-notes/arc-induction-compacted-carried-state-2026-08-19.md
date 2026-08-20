# Compacted carried state for the ARC induction tool loop

**Status:** DESIGN ONLY. No live-path file changes with this note. No test was run
for this note. Nothing under `results/` was written.

**Proposed requirement id:** REQ-ARC-WMTE-6540. The lineage is occupied through
6530, so 6540 is the next free id:

| id | what it is |
|---|---|
| 6500 | lean induce prompt with retrieval-backed exemplars |
| 6510 | vLLM backend for the scored path |
| 6520 | backend fork for the kernel pre-flight probe (2026-08-18) |
| 6530 | goal-probe samples the window tail, not only the head |
| 6540 | this note |

This note first proposed 6520, which was already taken. The table is here so the
next reader checks the lineage instead of assuming.

**Proposed env var:** `CARNOT_ARC_INDUCE_TOOL_COMPACT`. Unset or any value other
than `"1"` means byte-identical behaviour to today. The loop earns any default
flip through measurement, not assertion — same contract as the loop itself
(`arc_induction_tool_loop.py`, REQ-ARC-WMTE-6460).

---

## 1. The problem

The tool loop's `messages` list grows monotonically. Every assistant turn and
every full tool-result JSON stays in the conversation for the rest of the loop
(`arc_induction_tool_loop.py:504-526`). Two mitigations already exist and this
design must not duplicate or regress them:

| Already done | Where | What it bounds |
|---|---|---|
| `reasoning_content` dropped on feedback | loop line ~499 | think tokens never re-prefill |
| Per-turn `thinking_budget_tokens` = 3072 | loop `_think_budget` | decode per turn |
| Lean prompt (REQ-ARC-WMTE-6500, default OFF) | loop `_lean_prompt_k` | the BASE prompt size |

What is NOT bounded: the transcript. Each round appends the assistant tool-call
turn (an engine submission is 400–1,500 tokens of code), plus one tool result per
call. Worst-case tool results: `diff_grids` up to 200 cells (~3k tokens),
`query_region` up to 400 cells (~1k tokens), a mismatch report ~400–900 tokens.
Over 12 turns the transcript plausibly adds 15k–50k tokens on top of a 10k–17k
base prompt. These per-round figures are ESTIMATES from the tools' hard caps
(`MAX_DIFF_CELLS=200`, `MAX_REGION_CELLS=400`, `MAX_MISMATCHES_REPORTED=5`);
Phase 0 of the measurement plan replaces them with measured values.

Three context-length costs bind the scored run:

1. **Decode rate.** 42.4 tok/s at 10k context falls to 29.0 tok/s at 80k on the
   scored GPU. KV-cache growth is the mechanism.
2. **Concurrency.** KV per stream caps concurrent inductions. The agent framework
   starts one thread PER GAME with no pool.
3. **Queue wait.** Surplus requests queue against a fixed 2400s per-call timeout.
   A queued stream burns its timeout while generating nothing. The K=4
   concurrency probe timed out on prefill at 17k prompt tokens.

Shorter per-turn context moves all three. Section 9 says by how much, and marks
every number that is an estimate.

## 2. The core safety argument: dropped detail is recoverable

Every tool in `arc_induction_tools.py` is a deterministic pure function of the
fixed transition window. `diff_grids(4)` returns the same cells on turn 2 and on
turn 11. So a compaction that drops an old tool result does not destroy
information. It converts it into a re-fetchable fact at the cost of one tool
call. One tool call costs one prompt round of prefill, and prefill is ~15x
cheaper than decode (`arc_induction_tools.py` module docstring).

The only content compaction can destroy irrecoverably:

- The model's prose `content` on past assistant turns. This is usually empty on
  tool-call turns. The reasoning channel is already dropped today.
- The full source of superseded candidate engines. The carried state keeps their
  scores, a code fingerprint, and the best candidate's source verbatim.

Everything else — grids, diffs, mismatch reports, goal probes — is a view over
data the session still holds. That is the argument for why lossy summarization
does not cost accuracy in expectation. The measurement plan tests the claim
instead of trusting it (Section 11, gate G-Q).

## 3. What must be preserved across turns

| Item | Why the model cannot progress without it | Kept as |
|---|---|---|
| Base induction prompt (encoding rules, task spec, rendered transitions, tool instructions, repair seed note) | It defines the grid encoding and the output contract. Nothing else in the conversation restates it. | Verbatim, immutable, always message 0 |
| Best candidate code so far | The refinement target. Losing it forces a restart from a worse point. | Verbatim in carried state |
| Best candidate's scores | The number to beat. | In carried state |
| The LAST tool round (assistant turn + its tool results) | The report the model is about to act on. Summarizing it mid-thought breaks the local reasoning step. | Verbatim tail, never compacted |
| Per-candidate score history | Stops re-submission of refuted approaches. Feeds "am I converging". | Compact ledger rows |
| Which evidence was already fetched | Stops blind re-fetching. Lets the model cite facts it measured. | Bounded digests |
| Session-fixed facts (n_visible, held-out size, memorization coord scan active) | Cheap, tiny, orients the model. | One line in carried state |

What can be dropped, and why it is safe:

| Item | Why dropping is safe |
|---|---|
| Full JSON of old `diff_grids` / `query_region` results | Re-fetchable, deterministic, one call each. Digest keeps the headline (t, action, n_changed, bbox, value-pair histogram). |
| Full source of superseded candidates | Scores + fingerprint + first-mismatch line keep the refutation. `code_sha8` detects re-submission if this proves too lossy. |
| Old force-engine nudges and retry prompts | Loop-control text. The loop's counters live in Python variables, not in the transcript. |
| Old assistant `content` prose | Usually empty on tool turns. The carried state's candidate notes carry the mechanical equivalent. |

## 4. Carried-state schema

The carried state is ONE user message. The loop builds it mechanically from
`session.candidates`, the raw tool results, and the loop's own stats. No LLM
summarizes anything: a model-written summary would cost decode tokens, vary by
seed, and could hallucinate. Mechanical assembly is deterministic and unit-testable.

```json
{
  "v": 1,
  "kind": "arc_induce_carried_state",
  "turn": 7,
  "note": "Earlier turns were removed to keep this conversation short. This state is mechanical and complete. Do not re-derive what it records.",
  "session": {"n_visible": 22, "n_held_out": 3, "memorization_scan": true},
  "best": {
    "idx": 3,
    "code": "<verbatim best engine source>",
    "visible_mismatches": 2,
    "holdout_accuracy": 0.67,
    "is_memorizing": false
  },
  "candidates": [
    {"idx": 0, "visible_mismatches": 9, "holdout_accuracy": 0.0,
     "is_memorizing": false, "code_sha8": "a1b2c3d4",
     "code_head": "def engine(grid, action, data):  # gravity + push",
     "first_mismatch": "t=4 a=2: predicted no-op, real moved (7,3)->(8,3)"}
  ],
  "evidence": {
    "transitions_index": [{"t": 0, "action": 1, "changed": 4, "bbox": [7, 3, 8, 4]}],
    "diffs_fetched": [{"t": 4, "n_changed": 6, "value_pairs": {"0->5": 4, "5->0": 2}}],
    "regions_fetched": [{"t": 4, "which": "before", "r": [5, 9], "c": [2, 6]}],
    "goal_probes": [{"idx": 1, "n_grids": 24, "n_true": 0, "constant": true}]
  },
  "budget": {"tokens_est": 1480, "evicted": {"regions": 3, "diffs": 1, "candidates": 0}}
}
```

Field principles (Principle-Annotated Artifact Fields discipline):

| Field | principle |
|---|---|
| `v` | Schema changes must be visible to a reader of an old artifact; an unversioned blob silently drifts. |
| `best.code` (verbatim) | The monotone accept's floor. Truncating it would hand the model a broken refinement target and the loss would be silent. |
| `candidates[].code_sha8` | Detects the model re-submitting a refuted engine after compaction — the primary too-aggressive failure signal. |
| `candidates[].first_mismatch` | The one-line refutation. It is why the candidate lost, in the model's working vocabulary. |
| `evidence.*` digests | Distinguish "never looked" from "looked, found X". Without them the model either re-fetches (cost) or asserts from memory (hallucination risk). |
| `budget.evicted` | Makes eviction visible in the prompt itself; a silent eviction is the failure mode this design must not have. |
| `note` | Tells the model the history is gone on purpose. An unexplained amputated transcript invites the model to reconstruct it by guessing. |

## 5. Build and injection points

All anchors reference `python/carnot/agentic/arc_induction_tool_loop.py` at
today's HEAD.

1. **Evidence ledger.** A small `EvidenceLedger` object updates inside the
   per-tool-call dispatch block (next to `stats["tool_calls_total"] += 1`,
   line ~514). It digests each result dict at the moment it exists. It never
   re-parses the transcript.
2. **Trigger check.** At the top of the turn loop, before `_post_chat`
   (line ~478). The trigger reads the PREVIOUS response's measured prompt size:
   `usage.prompt_tokens`, with `timings.prompt_n` as the llama-server fallback
   (mirror of `_completion_tokens`, line ~231). Measured, not estimated —
   both llama-server and the vLLM backend (REQ-ARC-WMTE-6510) return it.
3. **Rebuild.** When the trigger fires, replace `messages` with:
   `[base message (verbatim)] + [carried-state user message] + [tail]`.
   The tail is the last COMPLETE round: the last assistant tool-call turn, its
   tool results, and any trailing user nudge.
4. **Then return to append-only** until the trigger fires again. Compaction is
   an event, not a per-turn rewrite. Expected events per 12-turn loop: 1–3.

**Conversation-validity invariant.** A `tool` message must always follow the
assistant message that carries its `tool_call_id`. The compaction unit is
therefore one complete round, atomically. A rebuild that orphans a tool message
is a bug; the chat template rejects it and `terminated_by=transport_error` makes
it visible (Section 10).

**Repair mode** (REQ-ARC-WMTE-6470) needs no special case. The seed engine and
its measured report live in the base message, which is immutable. The seed
occupies candidate ledger row 0 and carries into the `candidates` list like any
other row.

## 6. Bounds: caps and knobs

| Knob | Default | Meaning |
|---|---|---|
| `CARNOT_ARC_INDUCE_TOOL_COMPACT` | unset (OFF) | `"1"` enables. Anything else is byte-identical to today. |
| `CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH` | 8192 | Compact when measured prompt tokens ≥ turn-0 prompt tokens + this growth. |
| `CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET` | 2048 | Token budget for the carried-state message alone. |
| Tail size | fixed at 1 round | Not env-tunable, on the `MAX_RETRIES_PER_CALL` precedent: fewer knobs, and the tail's job needs exactly one round. |

The growth cap is relative to the measured turn-0 prompt, not absolute. The base
prompt varies 10k–17k by game and by lean-prompt mode, and an absolute cap would
either never fire on big bases or fire on turn 1 for small ones.

Sizing the carried state uses a chars/3 token estimate (conservative for
digit-heavy JSON). The estimate only sizes the state block. The TRIGGER uses the
server's measured count, so an estimate error shifts one block's size by a
bounded amount and can never unbound the context.

## 7. Eviction policy when the state budget binds

Drop in this order. Stop as soon as the state fits.

1. `regions_fetched` rows, oldest first. Coordinates only; cheapest to re-fetch.
2. `diffs_fetched` rows, oldest first. One call re-creates any of them.
3. `transitions_index` rows with `changed == 0`. Inert rows carry the least signal.
4. `candidates` middle rows. Always keep: row 0 (the first submission, and the
   seed in repair mode), the best row, and the last two rows.
5. **Never evict:** `best.code`, the `session` line, the schema envelope, and the
   verbatim tail round (which holds the latest mismatch report).

If the state still exceeds budget after step 4, inject it anyway and set
`compact_floor_hit: true` in stats. Fail toward completeness, visibly. Never
truncate `best.code` to fit — a truncated engine is worse than a long prompt.

## 8. Prefix-cache interaction

Both backends cache by longest common prefix: llama-server (`cache_prompt: true`,
already sent by `_post_chat`) per slot, vLLM by automatic prefix caching at block
granularity.

- Append-only turns re-prefill only the delta. That is today's behaviour and it
  is cheap.
- A compaction event rewrites everything AFTER the base message. The base itself
  stays a byte-stable prefix, so the cache still covers the 10k–17k base. The
  re-prefill cost per event is the carried state + tail: roughly 3k–5k tokens,
  ~1–3 events per loop. ESTIMATE.
- Between events the loop is append-only again, so cache benefits resume.
- At concurrency above the slot count, slot reuse across games already destroys
  the prefix cache today. A compacted transcript makes each cold re-prefill
  smaller. This is a second-order benefit; not counted in Section 9.

This is why the design compacts on threshold instead of rebuilding every turn.
A per-turn rebuild would pay the carried-state re-prefill 12 times and forfeit
the append-only cache between events for no additional context saving.

## 9. Which of the three costs this moves, and by roughly how much

| Cost | Moves? | Rough size | Measured or estimated |
|---|---|---|---|
| Decode rate on late turns | Yes | Context held near base+8k (~20–25k) instead of drifting to 40–60k. On the 42.4@10k → 29.0@80k curve that is ~10–20% faster decode on late turns, ~5–15% on loop wall-clock (early turns unchanged). | ESTIMATE — interpolated from two measured points; Phase 1 measures it |
| Concurrent streams | Yes, the largest lever | llama-server sizes KV per slot statically by context: a ~50% lower worst-case context permits ~2x slots at fixed VRAM. vLLM pages KV, so the win is proportional to actual KV held: ~40–60% less per stream at end of loop. | ESTIMATE — Phase 3 measures completions under K=4 |
| Queue wait against the 2400s timeout | Indirectly | Follows the slot count. More slots means fewer queued threads per the thread-per-game framework. | ESTIMATE — derivative of the concurrency result |

What this design does NOT move: the base prompt size (lean prompt
REQ-ARC-WMTE-6500 owns that) and decode tokens per turn (the think budget owns
that). The two levers compose: lean shrinks the floor, compaction bounds the
growth. Measure them separately before measuring them together.

## 10. Failure modes, and how each is visible rather than silent

| Failure | Mechanism | Visible signal |
|---|---|---|
| Model re-fetches evidence it already had | Digest too lossy | `refetch_tool_calls_post_compaction`: dispatch keys (tool, canonical args) seen before a compaction and repeated after it |
| Model re-submits a refuted engine | Candidate ledger too lossy | `duplicate_candidate_submissions` via `code_sha8` match |
| Parse rate shifts under the new prompt shape | Qwen3.8 reaches tool calls through the generic PEG autoparser, UNVERIFIED at design time; a changed prompt shape can plausibly change its behaviour | `tool_call_parse_failures` and `unparsed_tool_call_text_turns` already count it per run; the A/B compares arms (gate G-P) |
| Orphaned tool message after rebuild | Invariant bug | Chat template rejects the request; `terminated_by=transport_error` plus the unit test in Section 12 |
| Compaction thrash (fires every turn) | Trigger bug or growth cap below one round's size | `compactions` count per cell; expected ≤ 3, alarm above 5; `prompt_tokens_per_turn` shows a sawtooth |
| Carried state cannot fit its budget | Very long best engine | `compact_floor_hit: true`; the state ships whole rather than truncated |
| Quality quietly degrades | Sum of the above below alarm thresholds | Gate G-Q: paired holdout accuracy across arms; `mismatch_trajectory` convergence comparison |

## 11. Measurement plan

Reuse the existing 13-cell A/B harness shape (5 games x seeds, paired same-seed
cells). Every artifact declares `inference_substrate: live_llm_inference`,
`random_seed`, `reproducibility_checksum`, `model_specs`, and a PRECONDITIONS
step (server reachable, GGUF cached), per the Adversarial Artifact Verification
and Pre-Launch Preconditions disciplines. New stats fields carry `principle:`
annotations per the Principle-Annotated Artifact Fields discipline.

**Phase 0 — telemetry, no behaviour change.** Add `prompt_tokens_per_turn` to
`last_tool_loop_stats`, read from `usage.prompt_tokens` /` timings.prompt_n`.
Recording telemetry does not alter any request payload, so it is safe
unconditionally. Run the OFF arm once to replace Section 1's transcript-growth
estimates with measured values. If measured end-of-loop context rarely exceeds
~25k, the expected win shrinks and the operator should read Section 9 with that
number before funding Phase 1.

**Phase 1 — single-stream pilot A/B, 13 paired cells.** Arms: loop with
compaction OFF vs ON, same seeds, same windows.

**Phase 2 — quality non-inferiority at claim-grade N.** Only if Phase 1 passes.
N ≥ 30 paired cells (6 games x 5 seeds), per the Sample-Size Rigor rule (N ≥ 30
for any percentage-point delta claim). The documented ~40% A/A cell-divergence
floor under identical code means 13 cells detect only gross quality regressions;
13-cell quality numbers are pilot signals, never headline claims.

**Phase 3 — concurrency probe.** Repeat the K=4 probe that previously timed out
on prefill at 17k prompt tokens. Same games, same 2400s per-call timeout, both
arms.

Acceptance gates. Each is falsifiable and names its sample size.

| Gate | Metric | Pass condition | N and power note |
|---|---|---|---|
| G-M (mechanism) | p95 per-turn prompt tokens, ON arm | ≤ trigger + state budget + one round (~cap+5k) on every cell where the OFF arm exceeded the trigger | 13 cells. Near-deterministic given the caps; small N suffices. |
| G-P (parse safety) | pooled tool-call parse-failure rate | ON − OFF ≤ 5 percentage points; median `unparsed_tool_call_text_turns` per cell not worse | 13 cells yield ~300–500 tool calls per arm: ~80% power for a 5pp shift at α=0.05. A 2pp gate would need ~2,000 calls per arm (~50 cells); do not claim tighter than the data supports. |
| G-Q (quality non-inferiority) | best candidate holdout accuracy, paired | pooled ON−OFF ≥ −0.05 AND ON wins or ties on ≥ half the paired cells | Pilot at 13 cells; claim-grade at ≥ 30 cells (Phase 2). |
| G-W (single-stream cost) | median loop `wall_s` | ON ≤ 1.05 x OFF | 13 cells. |
| G-K (concurrency) | inductions completed inside the 2400s window at K=4 | ON ≥ OFF, and ON timeout count < OFF timeout count | One probe per arm, 4 streams. This gate is where the design earns its keep; if G-K shows nothing, the lever is not worth its complexity. |
| Advisory | `refetch_tool_calls_post_compaction` / total inspection calls | ≤ 20% | Advisory, tunable; a high value says the digests are too lossy, not that the design fails. |

**Kill condition.** If G-P fails — the parse rate collapses under the new prompt
shape — stop the program. That failure sits in the PEG autoparser, not in any
cap this design can retune.

**No default flip from this note.** A flip is a separate operator decision after
the gates, per the loop's own contract.

## 12. Tests the implementation must ship (Tests Must Run and Assert)

All with real assertions, none skipped, none touching tracked state
(`tmp_path` only, per the Test-Run Record Integrity discipline):

1. Env-unset pin: `CARNOT_ARC_INDUCE_TOOL_COMPACT` unset produces a
   byte-identical `messages` sequence on a scripted session (bomb-pin pattern
   from `test_arc_induction_tool_loop.py`).
2. Round-trip fidelity: carried state built from a synthetic session contains
   `best.code` verbatim, one ledger row per candidate, and the correct eviction
   counts.
3. Conversation validity: after a rebuild, every `tool` message's
   `tool_call_id` resolves to the preceding assistant turn. No orphan survives.
4. Eviction order: with a forced tiny budget, rows disappear in the Section 7
   order and `compact_floor_hit` sets when the floor binds.
5. Trigger arithmetic: compaction fires exactly when measured prompt tokens
   cross turn-0 + growth, and not on the turn after a rebuild.
6. Repair seeding: row 0 of `candidates` is the seed and survives eviction.

## 13. Out of scope

- Shrinking the base prompt (REQ-ARC-WMTE-6500 lean prompt owns it).
- Changing `reasoning_content` handling or the think budget (already done; this
  design must not regress either).
- Any change to the tools' report shapes or caps.
- A model-written summary step. Rejected above: costs decode, varies by seed,
  can hallucinate; the mechanical state is deterministic and testable.
- A worker pool for the thread-per-game framework. That is a separate,
  complementary fix to the queueing cost; this design only reduces each
  stream's footprint.

## 14. Cross-references

- `python/carnot/agentic/arc_induction_tool_loop.py` — the loop (REQ-ARC-WMTE-6460, 6470, 6500 hooks).
- `python/carnot/agentic/arc_induction_tools.py` — tool caps, holdout split, candidate ledger.
- `python/carnot/agentic/arc_recall_gated_resample.py` — repair-mode fire path; the ~40% A/A divergence floor.
- `openspec/capabilities/arc-world-model-trust-energy/spec.md` — REQ-ARC-WMTE-6400..6510 lineage; 6520 proposed here.
- `docs/research-notes/arc-induction-wall-consolidated-2026-08-12.md` — why induction levers need A/Bs with per-row data.
- CLAUDE.md: Simplified Technical English; Principle-Annotated Artifact Fields; Tests Must Run and Assert; Adversarial Artifact Verification + Sample-Size Rigor; never-prune.
