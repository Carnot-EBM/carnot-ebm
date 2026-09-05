# Bounded induction state, 2026-09-05

The mechanism is implemented on the existing live induction path and remains off.
No local-model efficacy claim, measured null, token saving, speedup, or score gain
has been established. No GPU run was started and no `results/**` file was written.

The ranked plan was: (1) retain candidate-specific failures grounded in observed
transitions; (2) retain whole prior source when it fits and compact deterministically;
(3) expose delivery receipts and the same policy in the offline twin; (4) run a
paired action-efficiency experiment only after operator authorization. Steps 1–3
are implemented. Step 4 is proposed below.

REQ-ARC-WMTE-7040 owns state in each E3 policy. It carries one generated candidate
and four distinct counterexamples across standard induce calls. Each failure
names the source SHA-256, transition SHA-256, action, wrong-cell count and up to
four predicted/observed cell pairs. These are refutations of that candidate on
that observation, not semantic proofs against a whole hypothesis family. Up to
eight current proposal rows are probed under the existing execution timeout guard.
Input grids and action data are copied. Malformed predictions become errors.
Game, level, grid shape and cell scale changes clear the state.

REQ-ARC-WMTE-7041 caps stored source at 32768 UTF-8 bytes and the added prompt
block at 4096 UTF-8 bytes. These are chosen design limits, not measured optimal
budgets. A long program is omitted whole, with its byte count and hash retained;
JSON records are never cut. Source omission loses the rule text, so the usefulness
of large-engine counterexamples to a 27B model is an open question. No model call
performs compaction. Raw, chat and optional vLLM generation reserve the added byte
count from the existing completion budget; insufficient space refuses locally.
This is a conservative byte-token assumption for the local tokenizer, not a
measurement of tokens or a cure for existing base-prompt truncation.

REQ-ARC-WMTE-7042 connects both plain and bounded policy induction, including the
level-up helper. The offline twin's explicit `--mechanism e3` delegates to
`E3AgentPolicy` and the existing eval runner. Scored liveness and eval diagnostics
record prepared calls, returned HTTP deliveries, added bytes, stored state,
compactions, resets, probe count and unsupported tool-loop successes. Chat
continuations count every returned request. Transport failures before a parsed
response are not counted as returned deliveries, even if a server received them.

The flag is `CARNOT_ARC_INDUCE_STATE_PERSISTENCE=1`; unset, zero and `true` remain
inactive. Its ledger entry is unevaluated and not promotable. An enabled run with
zero deliveries is an unexercised mechanism, not a measured null.

Evidence boundaries: memory reads the same proposal rows as the existing induce
prompt. It never loads a stored engine or holdout corpus. Plain induction already
exposes acceptance-tail rows unless `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1`; this work does
not repair that existing leak. Both proposed A/B arms must enable that flag.
Tool-loop state, refactor outputs, model-written summaries, disk restart recovery,
opaque reasoning and KV-cache continuation were skipped. They require different
interfaces or evidence. Refactor-produced source does not replace remembered
standard-induce source. Neither the planner nor the scored action budget changed.

The visible-state mechanism runs with the existing local Qwen3.8-27B proposer.
No Astra request or frontier dependency was added. Text and executable feedback
transfer mechanically to 27B; the ability to use them to save actions remains
unmeasured. Opaque provider reasoning state and provider-managed compaction do not
transfer through the present local completion API. This implementation does not
pretend that visible source is hidden reasoning state.

ARC Prize's [Astra article](https://arcprize.org/blog/astra) is dated 3 September
2026. It reports 62.7% for standard/max and 99.9% for provider/high; provider/max
is 98.6%. Its standard harness already passes visible notes. Its 3.66× and 49%
figures concern 167 matched game/reasoning pairs solved by both harnesses, across
public and semi-private tasks and reasoning levels. These are publisher-reported
measurements, not Carnot measurements. The comparison does not isolate compaction
from opaque reasoning-state preservation, and the headline scores use different
reasoning efforts.

Measured on CPU, with populations named:

- 40 new request-test cases passed. The regression selection contains those 40
  plus 101 existing cases in nine adjacent test modules: 141 passed total.
- 58 distinct final call-site mutations produced pytest exit 1 with a real
  assertion failure, then `cmp` exit 0 and restored pytest exit 0. No final
  mutant survived. The [mutation manifest](astra-induction-state-persistence-mutations-2026-09-05.json)
  records exact replacements, selectors, source hashes and assertion excerpts.
- One initial offline-dispatch mutation raised `KeyError`, so it did not qualify
  as assertion proof. An explicit consumer assertion replaced that accidental
  exception and the same deletion was rerun RED/restore/GREEN. An earlier empty-
  proposal trial was invalid because its restored baseline also failed: the
  existing prompt rejects empty evidence before memory is reached. That unused
  memory guard and its invalid test were removed. Neither trial is counted among
  the 58 accepted proofs. A redundant final record-pop compactor was also removed
  during review; four records with bounded palette values already fit after
  source omission. It was not claimed as a proven rule.
- One four-case scripted transport batch made 18 HTTP-shaped requests, including
  14 with prior state. No model ran. Each case reused ten synthetic transitions
  per attempt; eight eligible rows were probed on calls with retained source.
  The largest observed addition among those 14 blocks was 1699 UTF-8 bytes.
- One real offline `r11l` smoke used a 12-action budget, took 11 actions, completed
  zero levels and made zero generator calls. Induction was disabled. This is
  environment/entrypoint evidence, not a memory result.
- Final `pytest tests/ --collect-only` collected 61,776 cases but exited 2 on
  eight import errors. Four tests require unavailable `carnot._rust`; three import
  absent experiment modules 6814, 6819 and 6828; a quarantine test imports absent
  `exp4058_decentralization_moe_resume_build`. These tests are unchanged. No new
  skips or collection exclusions were added, and the whole suite is not green.
- Changed-file Ruff, formatting and spec coverage pass. The full spec audit
  reports 1,178 unreferenced tests across Rust/Python. The orphan-solver lint
  passes with 85 modules in its live closure. Whole-package mypy passes across 4,469 source files. The installed commit
  hooks are required for publication of this branch commit.

| Scripted case | Attempts / HTTP requests | Source UTF-8 bytes | State deliveries | Bytes per delivered addition | Stored refutations |
|---|---:|---:|---:|---:|---:|
| Small source, 2×2 grids | 2 | 119 | 1 | 1687 | 4 |
| Long source, 2×2 grids | 12 | 6121 | 11 | 1569 | 4 |
| Source over storage cap | 2 | 40121 | 1 | 407 | 0 |
| Small source, 64×64 grids | 2 | 119 | 1 | 1699 | 4 |

The 40,121-byte source leaves only omission metadata in this fresh memory; it
produces no new counterexamples. This is a limitation, not evidence of reuse.
The twelve-attempt case invoked 88 probes of a repeated window, not 88 distinct
observations. Requested output fell from 75,952 to 74,265 / 74,383 / 75,545 /
74,253 tokens respectively. Those numbers are request limits read from scripted
payloads, not generated or saved tokens.

Proof map: REQ-7040 covers exact opt-in, policy ownership, four scope keys and
both reset operations, first-call behavior, source storage/hash, guarded compile
and engine calls, eligible row limits, input copying, malformed output/metadata,
truthful failures, deduplication and publication. REQ-7041 covers sample/history
caps, whole-source compaction, prompt insertion, combined/split transport budgets,
completion reservation and refusal when output space is too small. REQ-7042
covers both policy bindings, default-off calls, legacy signatures, both bounded
helper variants, raw/chat/vLLM receipt accounting, chat continuations, tool-loop
exclusion receipts, scored/eval diagnostics and offline dispatch/budget/output.

The operator has not authorized the local-model A/B. It remains unfinished,
as do opaque reasoning, KV-cache reuse, tool/refactor integration and restart
persistence. No token/speed/action-efficiency benefit is inferred from CPU tests.
Session token usage is unavailable: the contract's `scripts/session-metrics.py`
is absent from this checkout, so no count was invented.

Reproduction uses the worktree interpreter and explicit paths:

```bash
PYTHONPATH=/home/ianblenke/carnot-wt-astra/python JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/python/test_arc_induction_state_persistence.py --no-cov --basetemp=/tmp/carnot-astra-state-persistence/reproduce -n 0 -q
PYTHONPATH=/home/ianblenke/carnot-wt-astra/python JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/ --collect-only --no-cov --basetemp=/tmp/carnot-astra-state-persistence/collect-reproduce -n 0 -q
```

Logs for this session are under `/tmp/carnot-astra-state-persistence/`. The
worktree needed an ignored `.venv` symlink to the shared environment. The CPU
smoke briefly used a local `environment_files` symlink, removed afterward.
The collection interlock flagged flag-ledger/E2E-plan edits authored while its
first run was active. Its marker had no attributed test writes. Both diffs were
reviewed and the marker retired by its documented intended-edit procedure; the
commit message records that decision. Final collection created no such marker.


Proposed operator decision: authorize a paired local Qwen3.8-27B A/B on `r11l`, `sb26` and `lp85`,
with generator seeds 0, 1 and 2 per game (nine pairs, eighteen runs), a 400-action
budget per run, `n_ctx=98304`, identical model hash, server settings and acceptance
splitting. Verify matching initial-frame hashes; the existing runner exposes no
environment-seed argument. Only the persistence flag varies; tool
induction stays off. Report all eighteen outcomes, including failures and zero-
delivery runs; use paired levels completed and actions to matched milestones as
primary endpoints. Secondary measures are returned deliveries, actual tokenizer
usage, wall time, prompt truncation and engine acceptance. Never count the designed
byte budget as saved tokens. The first r11l pair should verify
nonzero memory deliveries before the remaining predeclared pairs run. Keep the
flag off unless this controlled experiment supplies evidence. An operator must
authorize the GPU budget before this experiment starts.

The first commit attempt was refused by `eval-run-consumer-field-lint`: this
worktree lacks the ignored run-artifact directory. REQ-ARC-WMTE-6642 now exposes
the existing `--runs-dir` argument through `CARNOT_ARC_EVAL_RUNS_DIR`, for explicit
read-only selection of existing evidence. CLI selection wins; an empty environment
retains the default. The selected path prints, while source checks stay in this
worktree. Missing/empty corpora and un-emitted fields still fail. No applicable hook was
bypassed or weakened, and nothing was created under `results/`.

Seven new input-selection test cases plus seven existing lint cases pass (14
cases). Six additional call-site mutations prove environment delivery, empty-value
fallback, CLI precedence, evidence-path reporting, rejection on failure and
worktree source selection. All six produced assertion RED, `cmp` restoration and
GREEN; none survived. They are separate from the 58 induction mutations in the
manifest. The real-repository test now exercises the hook's CLI entrypoint with
the configured corpus. This is the sole addition to the original implementation
scope, needed to complete the requested commit without copying evidence.

Final evidence-path check read 15 JSON artifacts across the selected main-checkout
runs and the worktree flat eval artifact. It joined 476 distinct keys against
seven consumers and 184 producer files in this worktree and passed. Its selected
path was printed. Hook-input Ruff/format/spec checks and focused mypy pass.
The final full-tree collection still reports eight unchanged import errors,
now with 61,776 collected tests after the seven new hook-input cases.

The commit retry passed all applicable installed hooks. Post-commit reconciliation
confirmed fresh docs and reported one remaining issue: the existing global
spec-reference backlog (1,178 tests). That backlog and the eight whole-tree import
errors remain unresolved; the changed tests all carry requirement references.
