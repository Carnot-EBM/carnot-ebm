# Scope: enforced claims and wound-wait ordering for concurrent agents

Written 2026-08-29 after a session in which four agents worked one checkout and destroyed each
other's work six distinct ways. Prompted by Chroma's Fission protocol write-up
(trychroma.com/engineering/transactions), whose premise is the one this repository validated the
hard way: **traditional transactions assume a retry is cheap, and agent work is not.** Their
phrase is "minutes of latency and dollars of tokens." Ours is an agent that lost a full round of
edits while HEAD advanced eight commits.

This is a SCOPE, not an implementation. It sizes two changes and states honestly what each
cannot do.

## What exists today, measured

`scripts/harness_integrity_lint.py` implements DECLARATIONS, not claims:

| property | today |
|---|---|
| a session declares the paths it intends to touch | yes |
| that declaration constrains what THAT session may stage | yes (per-session since 2026-08-29) |
| that declaration stops ANOTHER session touching the same path | **no — zero cross-claim refusals** |
| declarations carry a timestamp | yes (`declared_at`) |
| conflicting declarations resolve deterministically | **no — first-come, and they deadlocked** |

The conductor side (`claimed_by_other_sessions`) only UNSTAGES another session's claimed paths
from its own commit. It never refuses, and it runs only in the conductor.

So a claim is advisory in exactly the direction that hurt: it binds the claimant and not the
stranger.

## The six failure mechanisms this is aimed at

Recorded in the concurrent-agents memory; four were discovered on 2026-08-29.

1. Work loss through the pre-commit stash window.
2. Silent RESURRECTION through the same window — an append verified against a stale reading
   produced a duplicate spec section, four headings where two belonged.
3. Shared index: one agent's `git add`/commit swallowed thirteen of another's staged paths.
4. `git commit -- <paths>` commits the WORKTREE, not the index — so the documented defence
   publishes vandalised content during the very race it defends against.
5. A stash race re-applied an in-flight MUTATION over a byte-identical restore, making two sound
   rules read as decorative. GREEN mutation verdicts became untrustworthy.
6. Two narrow scopes mutually deadlocked; fixed by per-session judging, but only by removing
   cross-session judging entirely rather than ordering it.

Enforced claims address 3 and 6 directly. Wound-wait addresses 6 properly. **Neither addresses
1, 2, 4 or 5** — those live in pre-commit's stash behaviour and in git's pathspec semantics, and
no claim protocol reaches them. Say so plainly rather than implying broader cover.

## Piece 1 — enforced claims

**Rule.** A path claimed by session A may not be STAGED by session B while A's declaration
stands. Today `check()` computes `claimed_by_other_sessions`-style information and does nothing
with it; this turns it into a refusal with a named owner.

**Where it binds.** Commit time, in the existing `harness-integrity-lint` hook. Not at edit
time — we do not control agents' file-writing tools, and a check that cannot see the write
cannot prevent it. This is a real limitation: two agents can still EDIT the same file
concurrently and only discover it when the second tries to commit.

**Cost.** Small. The comparison already exists; what is missing is the refusal branch, the owner
in the message, and the tests. Perhaps 60 lines plus a mutation proof.

**The failure mode to design against.** An abandoned declaration becomes a permanent lock on a
path nobody is working. Mitigation: declarations expire. A stale claim (say, older than 4 hours
with no commit touching its paths) is reported as stale and ignored rather than honoured, and
`--list` shows staleness. Fail toward NOT blocking, because a wrongly-held lock stops work while
a wrongly-released one only costs attribution.

## Piece 2 — wound-wait ordering

**Rule.** When two declarations conflict, resolve by `declared_at`, not by arrival:

- the OLDER declaration wins;
- the YOUNGER is *wounded* — its claim on the contested paths is revoked, it is told which
  session wounded it and why, and it re-declares;
- ties break on run id, so the outcome is deterministic and both sides compute the same answer.

**Why this and not locking-with-waiting.** Waiting deadlocks, and we hit that deadlock live
today. Wound-wait is deadlock-free by construction: a cycle cannot form because the order is
total and time-based.

**Where Fission's version differs from what we need.** Theirs wounds a transaction and treats
the abort as an EARLY COMMIT — the partial work stays. That only works because their writes are
prefix-safe: they measured 28.2% of modifications sharing a prefix or suffix, so a partial state
is still valid. Our equivalent property is the append-only/never-prune discipline, which is why
the resurrection incident was survivable. But we should not assume prefix-safety for code edits.
So: **wound the CLAIM, never the work.** A wounded agent keeps its edits and re-declares; it is
not asked to discard anything.

**Cost.** Small on top of piece 1 — an ordering comparison and the wound message. The
declarations already carry `declared_at`.

## What I would NOT build, and why

**Read-time locking.** Fission takes an exclusive lock at first page read. We cannot: agents read
through arbitrary tools we do not intercept. Pretending otherwise would produce a lock that
records intent and misses the actual access, which is the "guard that is green because it never
looks" class this repository already has too many of.

**A merge protocol.** Chroma tried git-based conflict resolution and abandoned it — **3 of 8
calls explicitly gave up** after repeated conflicts, agents skipping updates rather than merging.
Our agents did not give up, but only because they kept out-of-band snapshots. Building merge
tooling would be solving the wrong problem: the fix for a stash that reverts your worktree is not
better merging.

**Anything touching pre-commit's stash.** Mechanisms 1, 2 and 5 all originate there. That is the
single highest-value target in this whole area and it is NOT in this scope, because changing when
pre-commit stashes is a change to every commit in the repository and deserves its own proposal
with its own evidence. Flagged deliberately rather than bundled.

## Honest summary

Two cheap changes that close 2 of 6 measured mechanisms, with a clear statement that the other 4
are untouched. The larger prize — pre-commit's stash window — is named and deferred rather than
quietly folded in. If only one is built, build piece 1: the deadlock piece 2 solves has already
been worked around, while the shared-index collision it does not solve happened three times in
one day.
