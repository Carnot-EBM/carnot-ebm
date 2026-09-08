# A check for stuck probes: built, measured, and NOT shipped

**Date:** 2026-09-08. **Status:** rejected on evidence. Do not rebuild without reading this.

## The idea

The CUDA-probe defect fixed this morning (`c134c89c15`) had a signature that looked cheap to
detect corpus-wide. `native_llama_server_cuda_build` was **false in 6 artifacts and true in
zero**, on a host where the property was demonstrably true. A boolean that never takes one of its
values has never once disagreed with itself, and is indistinguishable from a working check.

So: scan every top-level boolean field in `results/experiment_*.json`, flag any field with enough
observations that never varies, and you have a generic stuck-probe detector. It would have caught
the CUDA probe without anyone looking.

## Why it does not work

Measured over **6,129 artifacts and 6,253 distinct top-level boolean fields**. With a floor of 30
observations, **35 fields never vary.** Of those, 33 are constant BY DESIGN:

    204  always False  submitted_to_leaderboard
    153  always False  scripts_research_conductor_modified
    100  always True   no_new_verifier_run
     60  always True   research_complete_yaml_parses
     51  always True   no_push
     46  always True   operator_only

These are **compliance assertions**: a task declaring what it did not do, or a health check that
is supposed to pass. `scripts_research_conductor_modified: False` on all 153 observations is the
discipline working, not a broken probe. A check that flags them fires on correct behaviour, and
CLAUDE.md is explicit that a check which cries wolf is worse than the gap it closes.

## The discriminator that nearly rescued it

The distinction is between a **capability probe**, which measures the host or the world, and a
**compliance assertion**, which declares what the agent did. Only the first kind can be stuck.
Filtering names on capability-ish words (`available|reachable|supported|enabled|cuda|gpu|build|
healthy|installed|loadable|present|detected`) reduces 35 to **2**:

    56  always True   polarfire_reachable
    48  always True   kv260_reachable

Both stuck at True, which is the mirror risk — a probe that may never have been able to report
failure. Both were checked and are benign: unreachability IS recorded in this project, through
the `honest_verdict` (`blocked_kv260_ssh_unreachable`), not through the boolean, and 110 artifacts
carry such a verdict. The booleans only exist on the success path, so their denominator is "runs
that got far enough to write them" — a population trap, not a lying field.

## The honest verdict

The signal is real but it does not separate the two classes without per-field knowledge, and the
name-based discriminator is a proxy for intent that any new field can defeat. **Prose, not a
check.** Recorded per the Error Lifecycle's step 6, which permits "no check" as an answer but not
silence.

**What DOES transfer, and is cheap:** when you ship a boolean probe, run it once against a
known-POSITIVE and once against a known-NEGATIVE. The CUDA probe survived for months because
nobody ever asked it a question whose answer they already knew. That is a one-line habit at
authoring time, not a corpus scan afterwards.

The scan itself is preserved in this note's commit message as a one-off audit tool. It is worth
re-running by hand if a stuck probe is ever suspected again; it is not worth wiring.
