# Dr. Zero: self-evolving search agents without training data (2026-10-24)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Sources read:

- Paper: Yue et al., "Dr. Zero: Self-Evolving Search Agents without Training Data", arXiv 2601.07055 (v2). Text extracted with `pdftotext` and read directly.
- Repository: https://github.com/facebookresearch/drzero (README only, read through a summarizing fetch tool, so the license and dependency lines are second-hand).

## What the method is

One base model is copied into two roles. The proposer writes a question and its answer. The solver answers the question with a multi-turn search tool. The solver trains with GRPO on the proposer's questions. The proposer trains on a reward computed from the solver's answers. The loop runs for three iterations.

The knowledge environment is a fixed search engine (E5-base over an English Wikipedia dump). No human question, answer, or demonstration is used.

### Proposer reward (paper Eq. 4)

Let n be the number of solver attempts and k the number that are correct.

- Reward is zero if k = 0 (too hard, or the answer is wrong) or k = n (trivial).
- Otherwise reward is (n - k) / (n - 1). It is highest when exactly one attempt is correct.
- A format reward is added.

The proposer prompt also requires each hop and the final answer to be supported by retrieved passages. The paper states this reduces wrong or ambiguous synthetic answers but does not remove them.

### HRPO (hop-grouped relative policy optimization)

Standard GRPO samples m questions per seed and n answers per question. That costs m(n + 1) rollouts (20 in their setup). HRPO samples one question per seed and five solver answers (6 rollouts). It sets the advantage baseline from other questions with the same hop count, not from sibling questions of the same seed. Hop count is only a grouping key. Difficulty still comes from the solver pass rate.

The paper reports that a single global baseline (REINFORCE++ style) was unstable on mixed question structures.

## What the paper measured

- Base models: Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct.
- Benchmarks: NQ, TriviaQA, PopQA, HotpotQA, 2WikiMQA, MuSiQue, Bamboogle. Metric is exact match.
- Against supervised Search-R1: the 3B model is higher on the three single-hop sets (reported gains of 22.9%, 6.5%, 18.4%). The 7B model reaches roughly 90% of Search-R1 on multi-hop sets and is higher on 2WikiMQA.
- Against the data-free baselines SQLM* and R-Zero* (augmented by the authors with search): higher on every task, with reported average gains of 39.9% and 27.3%.
- Iterations (Table 3, average EM): 3B goes 0.304, 0.319, 0.326. 7B goes 0.366, 0.372, 0.360.

## Limits the paper states or shows

- Gains saturate. The solver peaks within about 50 steps per iteration, and the second iteration is the last clear gain. The 7B model drops at iteration 3.
- The most common training failure was inconsistent token IDs across multi-turn search and reasoning steps. The 7B proposer hit it more often.
- The authors list entropy collapse in larger models, reward hacking, and bias amplification as future work. They did not test for them.
- The result holds for short-form, verifiable QA. The paper says its evaluation does not separate search gains from reading retrieved passages.
- Only a 3B and a 7B model were tested. No 27B result exists.

## Corrections to the earlier chat summary

The chat answer given before this note contained two claims that this reading does not support.

- "Up to 14.1%" came from a third-party aggregator. It does not appear in the paper text I read. Use the numbers above.
- "Matches or beats fully supervised agents" is too broad. The paper shows parity on some sets and about 90% on multi-hop sets for the 7B model.

## What could transfer to Carnot, and what could not

### Ideas worth a small test

1. **Pass-rate band as a filter, with no training.** Eq. 4 only needs a solver that can be run n times. Used as a filter (keep tasks with 0 < k < n), it needs no weight update. This fits the rule that excludes any change to the mandated 27B generator.
2. **Grouped baselines for the calibrated-decision selector.** HRPO normalizes within a structural group. If the `calibrated_decision` benchmark has a natural task-type axis, group-level baselines are a cheap variance control for a small selector. This is a design idea only.
3. **Curriculum from the solver's current ability.** The reward moves with the solver. A fixed difficulty ladder does not. This addresses the per-game transfer gap named in the AVO and generalization-floor rules.

### Where the analogy is weak

- **Where the answer comes from.** In Dr. Zero the proposer supplies the reference answer, grounded in retrieved text. An ARC-style proposer would need to supply a checkable solution that a game simulator can replay. Without that, "solvable" has no meaning. This is the main open design question.
- **Circularity.** The reward uses the solver's agreement with the proposer's own answer. A proposer and solver that share a blind spot can agree on a wrong answer. The Circularity / Oracle-Distinctness rule applies: any use must say what is independent of the solver. In an ARC setting the simulator replay can be that independent check. In text QA the retrieved passages play a weaker version of that role.
- **Scale and domain.** Nothing was shown at 27B, on grids, or with an offline-only runtime.
- **Live-path reachability.** A trained proposer or solver that the scored `E3AgentPolicy` cannot call produces no live capability. See the ARC Live-Path Reachability rule. A training-free filter has a better chance of being reachable than a new trained model.

### License and code reuse

The README states a non-commercial license (details in LICENSE.md, not read). Carnot is MIT-0. Do not copy repository code. Reimplementing from the paper's equations is the safe route. The code is also tied to a specific verl release (README lists verl 0.5.0). Run the untrusted-code audit before executing any of it.

## Not checked

- The text of LICENSE.md.
- Whether the repository contains reward-hacking or question-quality analysis beyond the paper.
- The compute used per iteration. The paper gives step counts (150 steps per model over 3 iterations) but this note does not convert them to GPU hours.
- Whether HRPO helps outside multi-hop QA. No source says it does.

## Suggested next step (not started)

Check whether the existing registry solvers plus the simulator replay can act as the "reference answer" that a game proposer would need. If they cannot, the proposer-solver idea has no oracle-distinct signal on ARC, and the note's idea 1 is the only part worth keeping.
