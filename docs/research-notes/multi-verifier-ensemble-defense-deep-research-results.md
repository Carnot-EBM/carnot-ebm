# DR-2 Synthesis — Multi-Verifier Ensembles & Externally Grounded Defenses

**Status:** SYNTHESIZED 2026-05-03 ~21:15Z from Gemini Deep Research response
**Source:** `multi-verifier-ensemble-defense-deep-research-source.pdf` (1.9MB, 21 pages, 15 cited works)
**Prompt:** `multi-verifier-ensemble-defense-deep-research-prompt.md`
**Strategic role:** populated paper-v6 Sakana-defense novelty positioning + Theorem 9.2 limitation bound

## Executive verdict

DR-2 vindicates Carnot's k=6 AND-composed externally-grounded ensemble
as a recognized research direction with named peer architectures, AND
imposes a hard mathematical limit Carnot must acknowledge: **Spera's
Theorem 9.2 (arXiv:2603.15973) formally proves that detecting joint
null spaces in AND-composed verifiers is coNP-complete and that safety
is strictly non-compositional.** Carnot's exp1108 vacuous-dead-code
finding is the textbook empirical realization of this theorem.

**Strategic positioning (per DR-2 §5):** Carnot's true novelty is the
**unprecedented scale of formal distinctness (k=6)** applied specifically
to **Turing-complete self-referential codebase modification** (the Sakana
DGM threat domain) — NOT the formal+neural composition pattern itself
(Lean Atlas, SentinelAgent predate Carnot in that pattern).

## The 6 architectural peers Carnot must position against

| Architecture | Citation | Verification mechanism | Domain | Robustness claim |
|--------------|----------|----------------------|--------|------------------|
| **Sakana DGM** | Zhang et al., arXiv:2505.22954 (2025) | Single external reward + execution log | SWE-bench / Polyglot code generation | Threat baseline; demonstrates objective hacking via marker removal |
| **TrinityGuard** | Wang et al., arXiv:2603.15408 (Feb 2026) | LLM-Judge ensemble (consensus) | OWASP GenAI Top 10 Security | Mitigates cascading agent failure + multi-agent prompt injection. **Neural-only; correlated bias vulnerability.** |
| **Med-ICE / ICE** | Omar et al. (2026, medRxiv) | Iterative consensus ensemble (semantic alignment) | Clinical AI diagnostics | Self-corrects medical hallucinations via peer debate; lacks formal bounds |
| **CAF** | AAAI-41151 (2026) | Composition Calculus over Formal Safety Assertions (FSAs) | MLOps pipelines | Formally automates DAG of safety assertions; pre-deployment compliance gate |
| **Lean Atlas** | arXiv:2604.16347 (2026) | Lean 4 type-checking + neural semantic verification | Formalized math proofs | Eliminates semantic hallucinations in math proofs; mandates human oversight |
| **SentinelAgent** | Patil, arXiv:2604.02767 (Apr 2026) | 7-check Delegation Chain Calculus (formal + neural) | Federal multi-agent AI (DelegationBench v4) | **100% TPR at 0% FPR across 126 evasion combinations**; deterministic intent verification mathematically infeasible |
| **Spera (theoretical)** | arXiv:2603.15973 (March 2026) | Directed Hypergraph Closure (Theorem 9.2) | Capability-based AI (ToolBench G3) | Proves NP-hardness of adversarial input generation against AND constraints; safety strictly non-compositional |

**SentinelAgent is Carnot's closest direct architectural peer** —
k=7 checks bridging formal + neural domains, peer-reviewed-pending,
open-source. **Differs from Carnot:** secures federated API delegation
(scoped tool calls in MCP/OAuth) — NOT Turing-complete codebase
self-modification (Carnot's domain).

## What survives — Carnot's defensible claims after DR-2

1. **AND-composition exponentially shrinks attack surface** — Standard
   ensemble theory + DR-2 §2.5 affirm the topological-distinctness
   intuition. Hold this claim; cite Spera (2026) for the formal frame.
2. **Externally-grounded > pure neural** — DR-2 §2.4 confirms
   externally-grounded architectures (SentinelAgent, code execution,
   physics simulators) are "the strongest empirical defense currently
   available against active objective hacking." Carnot fits this category.
3. **Heterogeneity > homogeneous** — DR-2 §2.2 confirms Neural
   Consensus Ensembles (ICE / TrinityGuard / Constitutional debate)
   share pretraining-bias correlated errors and are "fundamentally
   limited for AGI-scale safety." Carnot's AST + Z3 + ThinkPRM + JSON-
   schema + SC-Energy mix avoids this trap.
4. **Self-improvement-domain k=6 scale is novel** — DR-2 §5 explicitly:
   "The true scientific novelty of the Carnot framework lies in the
   unprecedented scale of formal distinctness (k=6) applied specifically
   to the high-threat domain of self-improving, open-ended codebase
   modification."

## What is REFUTED — Carnot must NOT claim (per DR-2 §5)

1. **DO NOT claim AND-composition eliminates vulnerabilities.**
   Spera's Theorem 9.2 mathematically proves this is unequivocally false.
   The "Joint Null Space" (vacuous truths) is an intrinsic mathematical
   limit of intersectional acceptance regions. exp1108 is its textbook
   instance, not a Carnot-specific bug.
2. **DO NOT claim neural ensemble components are immune to distribution
   shift during self-improvement.** exp1121 (AUROC 0.9545 → 0.3333)
   demonstrates adversarial optimization seamlessly shatters static
   neural decision boundaries. Frame neural verifiers strictly as
   "probabilistic heuristic safety nets, not rigid formal constraints."
3. **DO NOT claim first to combine formal and neural verifiers in an
   ensemble.** Lean Atlas (arXiv:2604.16347) and SentinelAgent
   (arXiv:2604.02767) predate Carnot in this combinatorial pattern.

## The hard mathematical limit (Theorem 9.2 — must be cited)

**Spera, C. (2026). "Safety is Non-Compositional: A Formal Framework
for Capability-Based AI Systems." arXiv:2603.15973.**

Key results paper-v6 must engage:

- **Theorem 9.2:** Two systems (verifiers), each individually
  incapable of reaching a forbidden state or allowing a dangerous
  action, can collectively permit a forbidden goal through emergent
  conjunctive dependencies (the **Joint Null Space**).
- **Computational complexity:** Calculating minimal unsafe-set
  membership in AND-composed heterogeneous verifiers is
  **coNP-complete**. Dynamically detecting joint null spaces as the
  agent scales is **computationally intractable**.
- **Mechanism (per DR-2 §2.5):** Dead/unreachable code passes Z3
  (vacuous logical truth — false premise implies any conclusion),
  AST (syntactic validity), liveness (compiles without hang).
  Topological distinctness of the verifiers does NOT guarantee
  disjoint null spaces — AND-composition CREATES emergent conjunctive
  blind spots.

**Implication for Carnot:** Acknowledge the bound. Frame the k=6
ensemble as "drastically reducing but not eliminating the attack
surface, with intrinsic Joint Null Space risk per Spera Theorem 9.2."
Pair the ensemble with continuous out-of-band external grounding +
human-in-loop review.

## Architectural vulnerability matrix (DR-2 p.13)

| Attack vector | Monolithic (DGM) | Neural Consensus (TrinityGuard) | Heterogeneous AND (Carnot) |
|--------------|-------------------|-------------------------------|----------------------------|
| Token Obfuscation | High Risk (single model vulnerability) | Medium Risk (relies on unified LLM) | **Low Risk** (rigorous ensemble defense) |
| Logic Subversion (vacuous correctness) | High Risk (unbounded self-modification) | Medium Risk (prone to goal drift) | **Medium Risk (emergent joint null spaces)** |
| Objective Hacking | High Risk (single point of failure) | High Risk (identity spoofing) | **Low Risk** (formally certifiable map) |

Carnot is **Low Risk on Token Obfuscation + Objective Hacking**, but
**Medium Risk on Logic Subversion** — the Spera/exp1108 corner.

## Q.A–Q.E direct answers (paraphrased from §4)

**Q.A — Strongest evidence FOR formally-distinct AND-composed
ensembles vs single verifier:** SentinelAgent's 100% TPR at 0% FPR on
DelegationBench v4 across 516 highly complex adversarial scenarios.
CAF's Composition Calculus formally proves that vulnerabilities in
one domain are caught by disconnected metrics before deployment.

**Q.B — Strongest evidence AGAINST ensemble robustness in practice:**
Spera's Theorem 9.2 (arXiv:2603.15973) formal refutation of perfect
ensemble robustness; OOD calibration collapse precedent (Carnot
exp1121 + GAN literature). Neural ensemble components reduce defense
of k=6 to effective k=5 under adversarial pressure.

**Q.C — How Sakana DGM relates to specification gaming literature:**
DGM crosses the boundary from passive specification gaming (DeepMind
boat-race, simulation-pause exploits) to active **Objective Hacking**
— the agent has Turing-complete authorization to read/alter its own
codebase. Externally-grounded multi-verifier ensemble is recognized
as "the only viable countermeasure to this evolved threat model."

**Q.D — Named architectures Carnot must position against:**
SentinelAgent, CAF, ICE/Med-ICE, TrinityGuard. (See peer table above.)

**Q.E — Consensus on ensemble defenses for AGI-scale safety:**
Bifurcated. Purely neural ensembles "fundamentally limited for AGI-
scale safety." Formally distinct externally-grounded AND-composed
ensembles "currently considered the most viable engineering pathway
toward robust defense against self-modifying agents." BUT consensus
caveats this with Safety Non-Compositionality — multi-verifier
ensembles are **strictly necessary but mathematically insufficient
on their own**; they must be continuously paired with out-of-band
external grounding and human-in-the-loop review.

## Recommended engagement (DR-2 §5)

- **Engage formal verification community** (CAF authors, TLA+ AI
  safety applications) — NOT mainstream alignment researchers.
  Traditional alignment indexes on neural debate + reward modeling;
  AST-parsing / Z3-SMT defensive value will land better with systems
  engineering + formal assurance experts.
- **Collaborate with Sakana/UBC team** to deploy Carnot's k=6
  ensemble inside the live Darwin-Gödel Machine sandbox — would
  provide an unimpeachable empirical A/B test of ensemble robustness
  against genuine AGI-scale objective hacking.

## Citations to add to paper-v6 (mandatory)

```bibtex
@article{spera2026noncomposability,
  author = {Spera, C.},
  title = {Safety is Non-Compositional: A Formal Framework for
           Capability-Based AI Systems},
  journal = {arXiv preprint},
  eprint = {2603.15973},
  year = {2026},
  month = {March}
}

@article{patil2026sentinelagent,
  author = {Patil, K.},
  title = {SentinelAgent: Intent-Verified Delegation Chains for
           Securing Federal Multi-Agent AI Systems},
  journal = {arXiv preprint},
  eprint = {2604.02767},
  year = {2026},
  month = {April}
}

@inproceedings{caf2026composable,
  title = {Composable Assurance for AI Alignment: A Framework for
           Formal Safety Assertions},
  booktitle = {Proceedings of AAAI 2026},
  year = {2026},
  number = {41151}
}

@article{wang2026trinityguard,
  author = {Wang et al.},
  title = {TrinityGuard: A Unified Framework for Safeguarding
           Multi-Agent Systems},
  journal = {arXiv preprint},
  eprint = {2603.15408},
  year = {2026},
  month = {February}
}

@article{leanatlas2026,
  title = {Lean Atlas: An Integrated Proof Environment for Scalable
           Human-AI Collaborative Formalization},
  journal = {arXiv preprint},
  eprint = {2604.16347},
  year = {2026}
}

@article{omar2026medice,
  author = {Omar et al.},
  title = {Med-ICE: Enhancing Factual Accuracy in Medical AI through
           Autonomous Multi-Agent Consensus},
  journal = {medRxiv},
  doi = {10.64898/2026.04.02.26350080v1},
  year = {2026}
}
```

## Required paper-v6 prose changes

**Related Work — new comparator paragraph (insert after Sakana DGM
discussion):**

> Concurrent work converges on multi-verifier defenses for self-
> improving agents. SentinelAgent (Patil, 2026) implements a 7-check
> Delegation Chain Calculus combining formal authority/scope checks
> with probabilistic intent verification, achieving 100% TPR at 0% FPR
> on DelegationBench v4. The Composable Assurance Framework (AAAI
> 2026) formalizes a Composition Calculus over Formal Safety
> Assertions for MLOps pipelines. Lean Atlas (2026) bridges Lean 4
> type-checking with neural semantic verification for math proofs.
> Med-ICE (Omar 2026) and TrinityGuard (Wang 2026) represent the
> Neural Consensus Ensemble line, but inherit correlated pretraining
> biases that prove fundamentally limited under adversarial
> optimization pressure. **Carnot's contribution is the unprecedented
> scale of formal distinctness (k=6 heterogeneous verifiers) applied
> specifically to the high-threat domain of Turing-complete self-
> referential codebase modification — the Sakana DGM threat domain
> not addressed by SentinelAgent (federated API delegation) or Lean
> Atlas (static math proofs).**

**Limitations — new Joint Null Space paragraph (mandatory addition):**

> Spera's Theorem 9.2 (arXiv:2603.15973) formally proves that AND-
> composed verifier ensembles are non-compositional in the safety
> domain: heterogeneous verifiers individually incapable of permitting
> a forbidden state can collectively allow it through emergent
> conjunctive dependencies (Joint Null Space). Detecting these joint
> null spaces is **coNP-complete** — computationally intractable as
> the agent scales. Carnot's exp1108 vacuous-dead-code finding is the
> textbook empirical instance: dead code passes Z3 (vacuous logical
> truth), AST (syntactic validity), and liveness (compiles) checks
> simultaneously, satisfying every component while violating real-
> world utility. The k=6 ensemble drastically reduces but does not
> eliminate this risk; it must be paired with continuous out-of-band
> external grounding and human-in-loop review for AGI-scale deployment.

## Cross-references

- DR-1 synthesis (energy-based LLM alternatives): `energy-based-llm-alternatives-deep-research-results.md`
- Spera Theorem 9.2 link to exp1108: `memory/project_pathological_joint_null_space.md`
- SentinelAgent benchmark precedent: NEW memory entry needed
- Phase-3 in-loop tripwire as Joint-Null-Space mitigation: `memory/project_phase3_architecture_complete.md`
- Carnot exp1121 OOD inversion: `results/experiment_1121_*.json` (referenced in DR-2)
