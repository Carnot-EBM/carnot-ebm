# Why the claim-audit reviewer fails (diagnosed 2026-10-08)

Scope: `scripts/experiment_claim_audit.py`, run by the conductor at each milestone close.
Nothing in this note is fixed yet. It records the cause, the evidence, and the options.

## Symptom

`ops/experiment_claim_audit_report.md` shows rows like
`reviewer call failed: OpenAI Codex v0.156.1 -------- workdir: ... model: gpt-6.1-sol ...`
with verdict `CANNOT_DETERMINE`. Across 352 artifact-by-report observations since 2026-09-26,
130 (37 percent) failed this way. Of the 44 reports committed since 2026-09-26, 30 contain
failures and 14 do not. Thirteen of the 14 clean reports fall in one window, from
2026-09-26 23:21 to 2026-09-29 21:06 UTC, and the other clean report is 2026-10-05 00:32.
Failures resumed in the report committed 2026-09-30 01:17 UTC. Earlier failures also appear
on 2026-09-05 and 2026-09-06 (codex 0.149.1) and on 2026-09-26 between 02:55 and 03:20 UTC
(codex 0.156.1), before the clean window. The cause of those earlier failures was not
investigated.

## Trigger: the 2026-09-29 model rename

The fallback model is set by `AGY_FALLBACK_CODEX_MODEL` in the service environment.

- Drop-in `96-model-gpt6sol-20260925.conf` set it to `gpt-6-sol`.
- Drop-in `97-model-gpt6.1sol-20260929.conf` (dated 2026-09-29 19:45 EDT) set it to
  `gpt-6.1-sol`. That rename was made in this project's outer-loop session.
- The stale `/usr/bin/codex` 0.156.1 accepts `gpt-6-sol` (tested: exit 0, answer `OK`) and
  rejects `gpt-6.1-sol` (HTTP 400). The clean window ended when the rename took effect.

So the stale binary was a hidden dependency that the old model name happened to satisfy. The
rename did not test the new model name against the binary the audit script actually runs.
The rename is the trigger. The stale binary and the `agy` failure are the causes.

## Cause: two faults in a row

1. **The primary reviewer, `agy`, returns nothing.** The conductor service sets
   `AGENT_TYPE_AUDIT=agy` and `AGENT_MODEL_AUDIT=gemini-3.8-flash-high`. The audit calls
   `agy --model ... --print <prompt>`. `agy` exits 0 with empty output and this on stderr:
   `jetski: no output produced - a tool required the "command" permission that headless mode
   cannot prompt for, so it was auto-denied.` The audit treats empty output as a failure.
2. **The fallback, `codex`, is the stale system binary.** After `agy` fails, the audit runs
   plain `codex exec --model gpt-6.1-sol -`. Under the service PATH
   (`.venv/bin:/usr/local/bin:/usr/bin`) that resolves to `/usr/bin/codex`, a pacman-installed
   codex-cli 0.156.1. It fails in 2.5 seconds with HTTP 400:
   `The 'gpt-6.1-sol' model is not supported when using Codex with a ChatGPT account.`
   The conductor avoids this with `CODEX_BIN=/home/ianblenke/.local/bin/codex` (drop-in
   `90-codex-bin-newer-20260910.conf`). The audit script does not read `CODEX_BIN`.

This is the same class of fault as the 2026-09-10 planner incident recorded in that drop-in.
An interactive shell has `~/.local/bin` on its PATH, so probes from a shell look healthy.

## Why the error text is never recorded

`_call` returns `stdout or stderr`. On success the answer is on stdout. On failure stdout is
empty, so it returns stderr. `codex exec` writes a banner and then an echo of the whole prompt
(about 65,000 characters) to stderr, and the real error comes after the echo. The report keeps
`review[:160]`, which is the banner. The conductor runs the audit with output inherited and
keeps no copy. So the cause was invisible for over a month.

## Evidence (reproduced 2026-10-08)

| Test | Result |
|---|---|
| `/usr/bin/codex` 0.156.1, tiny prompt, `gpt-6.1-sol` | exit 1 in 2.5 s, HTTP 400 above |
| `~/.local/bin/codex` 0.159.1, tiny prompt | exit 0 in 3.4 s, answer `OK` |
| `~/.local/bin/codex`, the real packet for `exp8248` (64,134 characters) | exit 0 in 38 s, valid verdict `NO_CLAIM`. The report had marked this same artifact as failed |
| `agy`, tiny prompt, 4 tries | 0 of 4 produced output |
| `agy`, the real packet for `exp8248` | exit 0, empty output, same permission message |
| `agy --mode plan`, tiny prompt, 4 tries | 1 of 4 produced output |
| `agy --sandbox`, tiny prompt, 4 tries | 0 of 4 produced output |
| `/usr/bin/codex` 0.156.1, tiny prompt, `gpt-6-sol` (the old fallback model) | exit 0 in 3.1 s, answer `OK` |

Size is not the cause. Failure rates by raw artifact size: under 100 KB 18 percent, 100 KB to
1 MB 40 percent, 1 to 10 MB 51 percent, 10 to 50 MB 55 percent. No artifact failed every time.

## What is not known

- Why the audit succeeded on about 63 percent of observations after the rename. The most
  likely route is `agy` answering without wanting a tool. Today `agy` fails every time I
  tried (0 of 4 tiny prompts, and the real packet), which would imply a higher failure rate
  than 37 percent. The `agy` binary was replaced on 2026-10-08 at 18:51, so today's
  behavior may differ from earlier runs. The earlier `agy` failure text is gone.
- Whether `agy` was already failing before the rename. The clean window of 2026-09-26 to
  2026-09-29 is consistent with a working `agy`, and also with a failing `agy` hidden by a
  working codex fallback. The data cannot separate the two.
- Whether the other audits (verifier authenticity, QA-layer, pages, ARC self-solve) launch
  bare `codex`. Their files carry a comment about the PATH problem, but I did not check each
  launch line. `scripts/artifact_convention_audit.py:151` has the same bare `codex` form.

## Consequence

`CANNOT_DETERMINE` is not a flagged verdict, so an unreviewed artifact creates no ledger row
and no escalation. About 37 percent of recent artifacts were never reviewed, and any real
finding among them is missing from `ops/audit-findings-ledger.md`. The audit reports success
because it wrote its receipt.

## Blast radius: six audits, not one

`AGY_FALLBACK_CODEX_MODEL` feeds seven scripts. Six of them launch plain `codex` in their
fallback (checked in the source): `experiment_claim_audit.py`, `artifact_convention_audit.py`,
`verifier_authenticity_audit.py`, `qa_layer_authenticity_audit.py`,
`pages_adversarial_audit.py` and `arc_self_solve_audit.py`. The seventh is the conductor
itself, which uses `CODEX_BIN` and is not affected.

Failed reviews in each audit's latest committed report on 2026-10-08 (units counted from
`## ` headings, so they are approximate):

| Audit | Failed reviews | Units |
|---|---|---|
| QA-layer authenticity | 20 | 21 |
| Artifact convention | 7 | 12 (9 `CANNOT_DETERMINE`) |
| Verifier authenticity | 15 | 56 |
| Experiment claim | 3 | 8 reviewed in the run |
| Landing page | none | no failure markers in this report |
| ARC self-solve | none marked | no "call failed" text; one codex banner appears and was not examined |

The QA-layer audit checks the guard code itself. Its rows in `ops/audit-findings-ledger.md`
by first-seen date: 9 on 2026-09-23, 17 on 09-24, 10 on 09-25, 12 on 09-26, 4 on 09-27,
4 on 09-28, 5 on 09-29, 4 on 09-30, and none since. The audit has produced no new findings
for 8 days. An empty ledger looks like a quiet period, not a blind audit.

## Options, cheapest first

1. **Honor `CODEX_BIN` in all six audits.** Use `os.environ.get("CODEX_BIN") or "codex"` where
   each script builds its command. This restores the fallback everywhere. Verified: the newer
   binary works on the failing packet. Do not revert `AGY_FALLBACK_CODEX_MODEL` instead. That
   variable also drives the conductor's own fallback, and the conductor uses the newer binary.
2. **Record the real error.** Keep the last 300 characters of stderr on failure, not the
   first 160, and name which backend answered. Without this the next failure is invisible too.
3. **Decide the primary reviewer.** Either fix `agy` for headless use or return the audit to
   codex. `--dangerously-skip-permissions` would let the reviewer run commands and is not
   recommended for a reviewer that must only read the packet. Routing is an operator decision.
4. **Make the stale binary impossible to hit.** Update or remove `/usr/bin/codex`, or add a
   start-up check that the `codex` on PATH accepts the configured model. This fault has now
   happened twice.
5. **Count unreviewed artifacts.** Have the audit exit with a visible warning when more than a
   set share of verdicts are `CANNOT_DETERMINE`, so a 37 percent blind spot cannot pass as a
   clean run.
