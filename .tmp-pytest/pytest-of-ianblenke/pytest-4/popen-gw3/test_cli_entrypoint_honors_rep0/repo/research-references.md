# Research References & Future Considerations

Existing notes.

<!-- EXP210_REFERENCES_START -->
## 2026-04-12 - Exp 210: Constraint Extraction for Instruction-Tuned Models

### Core papers
- **#1 Neuro-Symbolic Verification on Instruction Following of LLMs** (arXiv 2601.17789) - https://arxiv.org/abs/2601.17789
  Why it matters: Most direct fit to Carnot's current blocker: it treats instruction following as a constraint-satisfaction problem and combines logical plus semantic checks in one verifier.
  Carnot use: Use as the primary template for a prompt-to-constraint intermediate representation plus solver-backed verification path.
- **#2 ConstraintLLM: A Neuro-Symbolic Framework for Industrial-Level Constraint Programming** (EMNLP 2025) - https://aclanthology.org/2025.emnlp-main.809/
  Why it matters: Shows that instruction-tuned models can be specialized for constraint programming, paired with retrieval and guided self-correction, and evaluated on an industrial benchmark.
  Carnot use: Borrow the CP modeling pattern for scheduling and resource constraints, especially as a second solver route beside Z3.
- **#3 LLM Self-Correction with DeCRIM: Decompose, Critique, and Refine for Enhanced Following of Instructions with Multiple Constraints** (Findings of EMNLP 2024) - https://aclanthology.org/2024.findings-emnlp.458/
  Why it matters: Directly decomposes instructions into atomic constraints, then critiques failures at the constraint level using RealInstruct and IFEval.
  Carnot use: Build Carnot's first prompt-side atomic constraint extractor and use DeCRIM-style critique labels as supervision.
- **#4 CARE-STaR: Constraint-aware Self-taught Reasoner** (Findings of ACL 2025) - https://aclanthology.org/2025.findings-acl.1116/
  Why it matters: Separates easy versus ambiguous constraints and learns different reasoning traces for different constraint levels.
  Carnot use: Route high-confidence literal constraints to symbolic solvers and keep ambiguous constraints on a softer verification path.
- **#5 VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks** (arXiv 2511.04662) - https://arxiv.org/abs/2511.04662
  Why it matters: Formalizes each chain-of-thought step into first-order logic and checks whether each step is grounded in source context, commonsense, or prior steps.
  Carnot use: Prototype a typed step graph for arithmetic and logic traces instead of relying on raw free-form chain-of-thought text.
- **#6 PCRLLM: Proof-Carrying Reasoning with Large Language Models under Stepwise Logical Constraints** (arXiv 2511.08392) - https://arxiv.org/abs/2511.08392
  Why it matters: Constrains each reasoning step to explicit premises, rules, and conclusions so chain-level validation becomes possible even for black-box models.
  Carnot use: Adopt the premise-rule-conclusion record format for future step-level verifier experiments.
- **#7 Deductive Verification of Chain-of-Thought Reasoning** (NeurIPS 2023) - https://proceedings.neurips.cc/paper_files/paper/2023/hash/72393bd47a35f5b3bee4c609e7bba733-Abstract-Conference.html
  Why it matters: Still the clearest baseline for decomposing verification into small subprocesses with a constrained natural-language reasoning format.
  Carnot use: Use as the baseline to beat for stepwise validation and premise-minimization prompts.
- **#8 Faithful Logical Reasoning via Symbolic Chain-of-Thought** (ACL 2024) - https://arxiv.org/abs/2405.18357
  Why it matters: Translates natural language reasoning into symbolic expressions, then verifies both translation and reasoning with a verifier.
  Carnot use: Good bridge design for a prompt-answer pair where Carnot first converts text to symbolic state and only then verifies.
- **#9 Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning** (Findings of EMNLP 2023) - https://aclanthology.org/2023.findings-emnlp.248/
  Why it matters: Foundational hybrid pattern: translate to symbolic form, run deterministic inference, and use solver errors to refine the formalization.
  Carnot use: Keep as the minimum viable pattern for solver-backed extraction experiments.
- **#10 Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning** (arXiv 2510.01069) - https://arxiv.org/abs/2510.01069
  Why it matters: Provides a formal lens for mapping informal chain-of-thought into typed proof objects and treating well-typedness as a certificate of faithfulness.
  Carnot use: Use as design guidance for a typed intermediate representation rather than free-form regex over reasoning text.

### Benchmark assets
- **FollowBench: A Multi-level Fine-grained Constraints Following Benchmark for Large Language Models** (2024) - https://aclanthology.org/2024.acl-long.257/ - Seed a prompt-side benchmark with explicit content, situation, style, format, and example constraints.
- **CFBench: A Comprehensive Constraints-Following Benchmark for LLMs** (2025) - https://aclanthology.org/2025.acl-long.1581/ - Adds broader real-life constraint taxonomies and requirement-priority scoring.
- **RealInstruct** (2024) - https://aclanthology.org/2024.findings-emnlp.458/ - Real user multi-constraint instructions are a better supervision source than synthetic prompt lists.
- **VIFBench** (2026) - https://arxiv.org/abs/2601.17789 - Instruction-following verifier benchmark with fine-grained labels; closest external evaluation target to Carnot's gap.
- **IndusCP** (2025) - https://aclanthology.org/2025.emnlp-main.809/ - Industrial constraint-programming tasks for scheduling and resource-allocation extraction.
- **P-FOLIO** (2024) - https://arxiv.org/abs/2410.09207 - Human-written stepwise logical proofs for evaluating step-level reasoning extraction.
- **FormalBench** (2025) - https://aclanthology.org/2025.acl-long.1068/ - Program-semantics benchmark where formal specification inference is the task itself.
- **StructFlowBench** (2025) - https://aclanthology.org/2025.findings-acl.486/ - Useful if Carnot extends from single-turn extraction to multi-turn structural constraints.

### Monitorability and CoT risk evidence
- **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation** (2025) - https://arxiv.org/abs/2503.11926 - Shows chain-of-thought can help oversight, but strong optimization pressure can induce obfuscated reasoning.
- **Can Reasoning Models Obfuscate Reasoning? Stress-Testing Chain-of-Thought Monitorability** (2025) - https://arxiv.org/abs/2510.19851 - Explicit stress test showing some models can hide adversarial intent under obfuscation pressure.
- **Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity** (2025) - https://arxiv.org/abs/2510.27378 - Useful metric design: faithful traces can still be poor monitors when they omit crucial factors.
- **Diagnosing Pathological Chain-of-Thought in Reasoning Models** (2026) - https://arxiv.org/abs/2602.13904 - Gives concrete pathology categories and cheap diagnostics for post-hoc rationalization, encoded reasoning, and internalized reasoning.
- **Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?** (2026) - https://arxiv.org/abs/2603.22582 - Recent cross-model evidence that faithfulness varies sharply by family and hint type, which argues for model-specific monitorability gates.
<!-- EXP210_REFERENCES_END -->
