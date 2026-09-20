# JevBench and SemIf decision readout

Date: 2026-09-20

## What JevBench is

[JevBench](https://github.com/fstandhartinger/jevbench) evaluates typed decision
systems. Each task has a state, a bounded question, declared options, and a gold
answer. Systems can return one label or a probability distribution over the
declared options.

The local source was read at commit
`7ce310c7262ed49cc85853339a8a42459298e3f3`. Its public files contain 231
tasks. The split has 48 easy tasks, 111 hard tasks, and 72 original tasks. The
published v1.2 benchmark has 534 tasks. The remaining tasks are not present in
the public directory.

The repository license is MIT. This check used its local `LICENSE` file. The
evaluation script reads only data and documentation from that checkout. It does
not import or run the JevBench package.

## What SemIf is

SemIf is described by JevBench as an in-process option-logit decision system.
The JevBench adapter names the author's `semif_phase1.direct.score` path. It
says the path runs Qwen3.5-4B once per decision. It then applies a softmax to
the logits for the declared answer-letter tokens. The adapter records no
generated answer text.

This mechanism produces a distribution that is conditional on the supplied
options. It is not a probability over every possible answer. Target-workload
calibration is therefore still necessary.

The SemIf repository could not be fetched during this review. Its MIT license,
its independence from TypeSafe, and its README warning were not checked from
the SemIf source. The description above is supported by JevBench's adapter. It
is not an independent reproduction of SemIf.

## Exact scoring definitions

The following definitions were checked in `jevbench/metrics.py`,
`jevbench/composite_v12.py`, and `RESULTS-v1.2.md`.

Accuracy is argmax accuracy for classification tasks. JevBench uses the
expected value for ordinal tasks. The Carnot script uses argmax for every
declared option set because it evaluates option decisions, not JevBench's
separate ordinal rule.

Top-label expected calibration error uses 10 equal-width confidence bins. For
bin (b), let (n_b) be its row count. Let `acc_b` be its accuracy. Let
`conf_b` be its mean top-label confidence.

```text
ECE = sum_b (n_b / N) * abs(acc_b - conf_b)
```

Confidence 1.0 belongs to the last bin. JevBench clips confidence to the range
from 0 to 1 before binning.

Total-variation distance uses the exact declared label set.

```text
TVD(p, q) = 0.5 * sum_label abs(p_label - q_label)
```

The v1.2 calibration axis is for the hard tier. Its first component is:

```text
ece_component = max(0, 100 * (1 - ECE / 0.5))
```

Its fidelity component is:

```text
fidelity = 100 * (1 - mean_TVD)
```

The final calibration axis is the arithmetic mean of these two components.
ECE covers the hard tier. Fidelity covers only hard tasks with an exact gold
distribution. The full hard tier has 20 such tasks. The public hard file has
10.

The source returns only the ECE component when `mean_tvd` is absent. This is a
helper behavior. It does not change the published hard-tier definition, which
has gold-distribution tasks.

The Brier score is the multiclass sum:

```text
Brier = sum_label (p_label - one_hot_gold_label)^2
```

For two labels, this equals twice the usual single-probability binary Brier
score. JevBench reports Brier as a diagnostic. The v1.2 calibration axis does
not include it.

Label-only systems do not receive a measured calibration axis. The results
page says this counts as zero in the JevBench Score. The geometric score then
uses `max(axis, 1)` before taking a logarithm. Thus, the final composite uses a
floor of one point for this missing axis. Saying only that the composite uses
a literal zero is incomplete.

The public files are not the full benchmark. A score over these 231 tasks is
not a v1.2 leaderboard score. The Carnot artifact states this limit.

## Corrections to the review background

The calibration formula in the review background is correct. One scope detail
was missing. Published v1.2 calibration uses the hard tier. Its fidelity term
uses 20 probability tasks across the full hard tier. Only 10 are public.

The label-only statement needs one detail. Such systems have no measured
calibration axis and count as zero before composite scoring. The geometric
formula floors that value at one.

The 72 percent versus 21 percent option-order result is present in
`RESULTS-v1.2.md`. It applies to `open-alternative-jev` on yes/no
answer-judging tasks. It does not describe the SemIf row. The ranked run used
the author's order. The reversed order came from an adapter error and remained
only as raw evidence.

The results document states that costs are estimates for many open systems.
It also states that some latency values use an assumed adjustment. No
confidence interval was found in the three named scoring sources. This review
did not inspect every historical artifact for intervals.

## What maps to Carnot

The option-logit readout is a useful typed-decision input. Carnot can combine
it with an energy-based selector. The selector can accept, reject, or escalate
the proposed option. This advances the calibrated-decision floor without
claiming to reproduce a private TypeSafe method.

Carnot should report ECE and distribution fidelity next to Brier score. Brier
measures the full one-hot probability error. ECE measures confidence alignment.
Fidelity tests whether the full distribution matches an exact target. These
metrics expose different faults.

External tasks reduce self-authored circularity. JevBench is external to
Carnot. Its hard tasks are still authored and reviewed by the benchmark team.
They are not an independent ground-truth authority for every domain.

Low confidence can drive an escalation cascade. A calibrated selector can use
a fixed threshold. It can send uncertain cases to a larger model or a human.
Thresholds must be set on the target workload. They must not be selected on the
same rows used for the final claim.

Every finite-choice evaluation should include an option-order probe. The probe
must map both runs back to stable option IDs. It should report both accuracies
and the mean absolute probability shift per option.

Shared prompt prefixes can reduce repeated prefill work. A serving system can
reuse the state and question prefix before branching into option layouts. This
is a latency optimization. It must preserve the exact logits and option
mapping. The new evaluator does not implement shared-prefix caching.

The closest Carnot targets remain `GAP-ORACLE-DISTINCT` and
`GAP-DETECTOR-AUROC-4208` in `ops/verifier_gaps.md`. Both need a calibrated
selection policy. The JevBench tasks can provide an external evaluation input.
They do not replace held-out Carnot task data.

## Caveats

- The option probabilities depend on prompt wording, option text, option order,
  and tokenizer boundaries.
- One original-order score and one reversed-order score require two model
  evaluations per task. Each order still uses one forward pass.
- Single-token labels avoid generation and JSON parsing. They do not remove
  label-token priors.
- Temperature scaling changes confidence. It does not fix a wrong ranking.
- The script fits temperature out of fold. Its bootstrap intervals remain
  conditional on the public sample and the fixed model outputs.
- Ten public gold distributions are a small fidelity sample. The interval must
  stay beside the point estimate.
- ECE depends on its bin rule. A 10-bin estimate can be unstable on small
  splits.
- The public files were selected by the benchmark authors. No result from them
  supports a claim about the held-out tasks.
- JevBench authors report their own benchmark results. Many costs are
  estimates. Some latency values are adjusted assumptions.

## Verification record

The following claims were checked directly against the local JevBench source:

- The task schema and the three public split sizes.
- The source commit and SHA-256 values for the public files.
- The 10-bin top-label ECE implementation.
- The multiclass Brier definition.
- The total-variation definition.
- The hard-tier calibration formula and its 20 full-tier probability items.
- The label-only treatment in the v1.2 score.
- The 72 percent versus 21 percent option-order report and its system name.
- The 534-task full benchmark count.
- The estimated-cost and adjusted-latency disclosures.
- The absence of confidence-interval text in the three named scoring sources.
- The JevBench repository's MIT license.

The following claims were not verified from their primary source:

- SemIf's repository license and its independence from TypeSafe.
- The exact text of SemIf's README warning.
- SemIf's internal implementation beyond the JevBench adapter description.
- TypeSafe's private training method or calibration process.
- The accuracy, cost, and latency claims through a new live reproduction.
- Any result on JevBench's held-out tasks.
- The absence of confidence intervals from all related repositories and
  historical artifacts.

No live model ran during this review. No GPU result is claimed.

## Live result, 2026-09-20 (outer loop, added after codex's draft)

**Run.** `scripts/jevbench_readout_eval.py` scored the 231 public JevBench tasks with our
local Qwen3.8-27B (Q4_K_M GGUF, llama.cpp, one RTX 3090). It reads the option logits in one
forward pass, zero-shot, with no chat template. Every task ran in both option orders
(462 calls). The whole run took 377 s including model load and the bootstrap. Artifact:
`results/jevbench_readout_eval_2026_09_20.json`. Seed 42, 5 folds, 1000 bootstrap resamples.
The live positive control passed first (8 of 8 trivial questions, mean probability of the
right option 0.99).

| Split | n | Accuracy (95% CI) | Top-label ECE | Brier |
|---|---:|---|---:|---:|
| easy | 48 | 1.000 | 0.006 | 0.000 |
| original | 72 | 0.958 (0.903-1.000) | 0.098 | 0.091 |
| hard | 111 | 0.739 (0.658-0.820) | 0.076 | 0.323 |
| all | 231 | 0.861 (0.818-0.905) | 0.060 | 0.184 |

- **Hard-tier calibration.** The JevBench-style score is 80.3 raw (CI 67.9-86.4), from only
  the 10 public gold-distribution items. The full benchmark uses 20. Mean TV distance 0.242
  (CI 0.100-0.399).
- **Temperature scaling did not help.** K-fold temperatures were 0.82-0.90, and overall ECE
  went from 0.060 to 0.062. The raw readout was already about as calibrated as this data
  can show.
- **Option order matters on hard items.** Reversing the options dropped overall accuracy from
  0.861 to 0.810 and hard-tier accuracy from 0.739 to 0.658. Mean absolute probability shift
  was 0.070 overall and 0.109 on the hard tier. Easy items did not move.
- **Cost.** About 0.8 s per decision on a 27B model, with no prefix reuse and no batching.

**Correction: the first full run was invalid and is not kept.** The scorer read the logits
row `scores[-1]`. That is the last row of the whole context buffer, which is never written,
not the last prompt token's row. Every readout was therefore uniform. That run scored at
chance (about 30% accuracy) and the option-order shift was exactly 0.0 on all 231 tasks. The
artifact still said "complete_live_option_readout_evaluation". codex's stub tests could not
catch it. The fix reads `scores[len(tokens) - 1]`. The script now also runs a known-answer
positive control before any benchmark, and refuses to write an artifact when more than half
of the distributions are uniform. Lesson: a null-looking result from a new readout needs a
positive control before it is read as a finding.

**How to read the numbers.** They are a health check of the readout on our model, not a
leaderboard entry. They are not comparable to SemIf's published axes: this is the 231-task
public subset, the accuracy is not JevBench's Intelligence axis, the calibration uses 10
gold items instead of 20, the model is 27B instead of 4B, and there is one run with no
repeat. What the run does show: option-logit readout on our local model gives usable,
reasonably calibrated probabilities on text decisions, and it is order-sensitive on hard
items. It shows nothing about ARC grid-state decisions.
