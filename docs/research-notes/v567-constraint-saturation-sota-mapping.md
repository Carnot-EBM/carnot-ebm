# V567 Constraint Saturation SOTA Mapping

Planning date: 2026-08-23.

This note records primary sources checked during Exp6555. The fixture uses exact DRIFT constraints, Z3 receipts, and deterministic clause checkers. It uses no model output.

## Source Rows

- 2608.12426: Large Language Models Can Follow Instructions, But Not Many at Once: Phase Transitions in Compositional Constraint Satisfaction
  - URL: https://arxiv.org/abs/2608.12426
  - PDF: https://arxiv.org/pdf/2608.12426
  - Checked: 2026-08-23T17:04:27Z; direct arXiv available: True
  - Failure mode: Per-clause accuracy can remain high while joint success collapses.
- 2602.13217: VeRA: Verified Reasoning Data Augmentation at Scale
  - URL: https://arxiv.org/abs/2602.13217
  - PDF: https://arxiv.org/pdf/2602.13217
  - Checked: 2026-08-23T17:04:27Z; direct arXiv available: True
  - Failure mode: A paraphrase can drift semantically unless executable constraints, not prose, define the label.
- 2606.19808: Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning
  - URL: https://arxiv.org/abs/2606.19808
  - PDF: https://arxiv.org/pdf/2606.19808
  - Checked: 2026-08-23T17:04:27Z; direct arXiv available: True
  - Failure mode: A route can spend more compute or flip a correct answer while looking good on recovered failures.
- 2608.14569: Position: Certified Correctness in Neural Constraint Reasoning Requires Symbolic Integration
  - URL: https://arxiv.org/abs/2608.14569
  - PDF: https://arxiv.org/pdf/2608.14569
  - Checked: 2026-08-23T17:04:27Z; direct arXiv available: True
  - Failure mode: A confident neural verifier can violate hard constraints under distribution shift.
- 2608.18921: SMTrap: Cost-Effective DoS Attacks Against Large Reasoning Models via SMT Conflict Guidance
  - URL: https://arxiv.org/abs/2608.18921
  - PDF: https://arxiv.org/pdf/2608.18921
  - Checked: 2026-08-23T17:04:27Z; direct arXiv available: True
  - Failure mode: Solver conflict count can become a false proxy for model difficulty.

## Method Mapping

- compositional_constraint_saturation: Freeze DRIFT variants across counts 1-12 and score per-clause plus joint success with deterministic checkers.
  - Falsifiable use: A later Exp6556 row must show the all-clause phase curve by count without aggregate-only success.
  - Failure mode: Per-clause accuracy can remain high while joint success collapses.
- executable_equivalent_and_hardened_variants: Use executable DRIFT constraints to build equivalent surfaces and hardened rows before model outcomes exist.
  - Falsifiable use: Equivalent rows must keep the source constraint hash, and hardened rows must equal source plus one declared clause.
  - Failure mode: A paraphrase can drift semantically unless executable constraints, not prose, define the label.
- selective_verification_budget_control: Require a longer-flat control and harmful-intervention counts before Exp6556 can credit decomposition or routing.
  - Falsifiable use: Fixture rows expose count, surface, timeout, and censoring fields needed to charge selective interventions.
  - Failure mode: A route can spend more compute or flip a correct answer while looking good on recovered failures.
- symbolic_certification: Keep Z3 and executable clause checkers as the only fixture authority; learned components may not certify labels.
  - Falsifiable use: Every row must round-trip through deterministic per-clause and joint checkers.
  - Failure mode: A confident neural verifier can violate hard constraints under distribution shift.
- solver_guided_constraint_stress: Record solver effort and interaction class separately from correctness labels.
  - Falsifiable use: Later rows can test whether interaction and solver effort predict cost without becoming label authority.
  - Failure mode: Solver conflict count can become a false proxy for model difficulty.

## Bottom-line fixture contract

- Lineages: 36.
- Constraint load counts: 1 through 12.
- Variant modes: equivalent and hardened.
- Surface forms: brief and table.
- Release authority: Z3 plus executable per-clause and joint checkers.
- Downstream models may be measured against these labels. They may not create labels.
