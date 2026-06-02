# Research Studying — Ranked Ideas for Future Experiments

Existing queue.

<!-- EXP210_STUDYING_START -->
## Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models

### Ranking update
| Rank | Idea | Score | Why it matters |
|------|------|-------|----------------|
| 1 | Prompt-to-constraint intermediate representation with solver fallback | 625 | NSVIF, DeCRIM, and ConstraintLLM all point to the same fix: extract atomic constraints from the instruction before verifying the answer. |
| 2 | Benchmark-first extraction workbench | 500 | FollowBench, CFBench, RealInstruct, and VIFBench provide the missing datasets needed to measure extraction recall and false positives directly. |
| 3 | Dual-path verification: prompt-answer first, CoT second | 500 | CoT verification is promising, but monitorability papers say Carnot should never depend on raw CoT alone. |
| 4 | Typed step-graph verification for arithmetic and logic traces | 375 | VeriCoT, PCRLLM, Deductive Verification, and Typed CoT all support moving from free-form traces to explicit premises and rules. |
| 5 | Constraint-programming route for scheduling and resource tasks | 240 | ConstraintLLM plus IndusCP is the best external path for Carnot's scheduling extractor gap. |
| 6 | CoT monitorability score and fallback policy | 240 | Recent monitorability work implies Carnot needs a gate deciding when CoT evidence is safe to trust. |

### Key takeaways
- The strongest direct fit is prompt-side instruction verification: convert instructions into atomic constraints first, then verify the answer against them.
- Step-level CoT verification is now technically credible, but only when reasoning traces are reformatted into explicit premises, rules, and typed steps.
- Benchmark coverage for fine-grained instruction constraints is finally good enough to evaluate extraction quality directly instead of using answer accuracy as a proxy.
- Recent monitorability papers make raw chain-of-thought an unsafe sole source of truth; Carnot needs a fallback path that does not trust CoT by default.

### Proposed experiments for 2026-04-15
- **EXP-211 - Instruction-to-Constraint IR Benchmark**
  Goal: Build a gold benchmark of atomic prompt constraints from FollowBench, RealInstruct, CFBench, and VIFBench, then measure extraction recall and false positives on instruction-tuned models.
  Hypothesis: Prompt-side decomposition will reduce false positives more than answer-only regex extraction because the verifier will know exactly which constraints matter before inspecting the response.
  Success criteria: Atomic constraint recall >= 0.85 on the curated benchmark, satisfied-constraint false-positive rate <= 0.05, and measurable improvement over the current regex plus Z3 promptless path.
- **EXP-212 - Dual-Path CoT Verifier with Typed Step Graphs**
  Goal: Implement a step-level verifier for arithmetic and logic traces using premise-rule-conclusion records inspired by VeriCoT, PCRLLM, Deductive Verification, and Typed CoT.
  Hypothesis: A typed step graph will catch errors that answer-only checking misses, but only when combined with prompt-derived constraints and a fallback to answer-level verification.
  Success criteria: On a live instruction-tuned cohort, catch >= 25% of wrong answers missed by prompt-only verification while adding < 2% extra false positives on correct answers.
- **EXP-213 - CoT Monitorability Audit and Fallback Policy**
  Goal: Measure whether Qwen and Gemma instruction-tuned models expose enough faithful reasoning to justify CoT-based extraction, using recent faithfulness and pathology metrics.
  Hypothesis: Monitorability differs by model family and task, so Carnot should gate CoT extraction behind a measured trust score rather than assuming traces are faithful.
  Success criteria: Produce a per-model monitorability score, a pathology breakdown, and a simple policy that predicts when to trust CoT extraction versus prompt-answer-only verification.
<!-- EXP210_STUDYING_END -->
