# Google Deep Research — Externally-Grounded Multi-Verifier Ensembles in Agent Safety

**Status:** PROMPT DRAFT — for paste into Gemini chat UI Deep Research
**Drafted:** 2026-05-03 ~22:00Z
**Strategic role:** sharpen paper-v6's Sakana-defense novelty claim with comparative literature
**Predecessors:** DR-1 (energy-based LLM alternatives survey, 2026-05-03 ~21:30Z)
**Output target:** comprehensive landscape of multi-verifier ensemble defenses in AI/agent safety

---

## Paste boundaries

```
START:  "## The Deep Research request" (line ~22)
END:    end of "## Output format requested" section
SKIP:   this header, "## Why this prompt now"
```

---

## The Deep Research request

I am surveying the literature for a research project that defines its core
architectural contribution as an open-source, **externally-grounded
multi-verifier ensemble defense against reward hacking in self-improving
AI systems.** The framework uses a k-AND-composed ensemble of formally
distinct verifiers — meaning a model output is accepted only if ALL
k verifiers (each operating on a different formal or semantic
discipline) independently grade it as valid. The verifiers themselves
are NOT trained jointly with the model being improved, so the model
cannot game them by modifying their parameters.

I need a comprehensive survey of the comparative literature so I can
position this contribution honestly against prior and concurrent work.

Please produce a structured research report on **multi-verifier
ensembles, ensemble-based safety mechanisms, and externally-grounded
reward signals for AI agents**, focused on work published 2022-2026.

### Context: the threat model under defense

The open problem this addresses, articulated most concretely in the
Sakana Darwin-Gödel Machine paper (Zhang et al., arXiv:2505.22954, May
2025): self-improving agents will REMOVE the markers used to detect
their own undesired behaviors when those markers conflict with the
agent's improvement objective. The Sakana team explicitly observed
their agent removing the hallucination-detection markers in the
hallucination-detection reward function. Single-evaluator self-
improvement is fundamentally vulnerable to this attack pattern.

The defense being proposed is: REPLACE the single (gameable) evaluator
with an ensemble of formally-distinct verifiers, AND-composed (an output
must pass ALL of them), such that gaming requires the agent to find
inputs that satisfy ALL the formal disciplines simultaneously while
violating ground truth. The architectural intuition is that the
intersection of formal-discipline acceptance regions is much smaller
than any single verifier's acceptance region, making attack-crafting
much harder.

### Survey scope (in priority order)

**1. Multi-verifier ensembles in formal verification literature**

Theoretical and empirical work on:
- AND-composition of formal verifiers (e.g., SMT solvers, model
  checkers, type checkers, runtime monitors)
- Joint null spaces / shared blind spots across formally distinct
  verifiers
- Soundness / completeness composition theorems for ensembles
- The "vacuous correctness" problem: outputs that pass all formal
  checks but fail real-world semantics
- Sample applications: program verification, hardware verification,
  protocol verification

**2. Ensemble methods in AI safety / agent alignment**

Specifically work that uses MULTIPLE evaluators to constrain a single
agent's behavior. Include:
- Constitutional AI (Anthropic) and successors
- AI Safety via Debate (Irving, Christiano, Amodei)
- Multi-reward-model RLHF
- Reward modeling robustness papers
- Adversarial debate / cross-examination protocols
- The "outer loop / inner loop" architecture in RLHF

**3. Reward hacking defense literature post-2023**

Work on defending against agent gaming of evaluators:
- Reward function shaping, auxiliary loss designs
- Adversarial robustness of reward models
- "Specification gaming" literature (DeepMind, OpenAI)
- Goodhart's law manifestations in RL reward models
- Sakana DGM (arXiv:2505.22954) and any follow-on / response papers
- 2024-2026 literature on evaluator-gaming attack/defense

**4. Theoretical results on ensemble robustness**

Mathematical work that bounds:
- How AND-composition of N independent classifiers reduces false
  positive rate
- When ensemble independence FAILS (correlated errors, shared training
  distribution biases)
- NP-hardness or PSPACE-hardness results for adversarial input
  generation against ensembles
- Information-theoretic limits on ensemble robustness

**5. Externally-grounded reward signals**

Work that uses signals from outside the agent's training distribution:
- Tool-use as ground truth (e.g., code execution, search results)
- Multi-modal grounding (vision, embodied feedback)
- Symbolic verification of LLM outputs
- "Process reward models" vs "outcome reward models"

### What I already cite or know well (deprioritize)

Surface only if there are recent developments I might have missed:

- Sakana Darwin-Gödel Machine (Zhang et al., arXiv:2505.22954, May 2025)
- Constitutional AI (Bai et al., 2022) — original Anthropic work
- BEAVER-lite (arXiv:2512.05439) — prefix-closed deterministic LLM verifier
- Goodfire's white-box neuron inspection (complementary, not multi-verifier)
- Themesis SeedIQ (refused code release, sacrificing prize)
- Z3 SMT solver, AST analysis libraries (the formal-verifier components,
  not the ensemble methodology)
- Standard ML ensemble theory (boosting, bagging, random forests) —
  not directly relevant unless applied to safety
- Process Reward Models (PRM) lineage including ThinkPRM

### Specifically Carnot's verifier ensemble (concrete commitment under test)

The framework AND-composes a k=6 ensemble of formally distinct verifiers:

1. **Z3 SMT solver** — formal correctness for arithmetic / SAT-style constraints
2. **AST structural verifier** — syntactic / structural validity of code outputs
3. **Semantic embedding probe** — distance-in-embedding between output and
   natural-language description
4. **ThinkPRM probe** — process reward model checking step-by-step reasoning
5. **JSON schema validator** — output well-formedness against declared schema
6. **SC-Energy (Set-Consistency Energy Network)** — set-level consistency check

Two empirical findings the literature should help interpret:

- **exp1108 (vacuous-dead-code joint null space):** {Z3, AST, liveness}
  share a null mode where vacuous unreachable code passes all three.
  Topological distinctness of formal verifiers does NOT guarantee
  disjoint null spaces.
- **exp1121 (SOSKANEnergyV3 OOD inversion):** a single verifier reached
  AUROC=0.9545 on its training corpus but AUROC=0.3333 (worse than
  random) on a production corpus. Verifier calibration is non-stationary
  across distributions.

### What I want to learn

For each architecture, technique, or theoretical result surfaced, I need:

1. **What it claims** — the ensemble or grounding architecture in one sentence
2. **What it demonstrates** — empirical scale, benchmarks, attack scenarios
3. **Primary reference** — citation with arXiv ID, conference, year
4. **Lineage** — which earlier work it builds on
5. **Open-source status** — code/weights public? license?
6. **Comparison to single-evaluator baseline** — does the ensemble actually do
   better than a strong single evaluator? On what attack surface?

I particularly want to find:

- **Recent 2024-2026 work** I likely haven't seen
- **Negative or null results** — papers reporting that multi-verifier
  ensembles failed to provide the expected robustness, or that joint
  null spaces dominated the security analysis
- **Theoretical results** — papers proving AND-composition bounds, NP-hardness
  for adversarial crafting, or limits on ensemble robustness
- **Industrial deployments** — major labs (Anthropic, OpenAI, Google,
  Apple) using multi-verifier ensembles in production safety systems

### Specific questions to answer

**Q.A — What is the strongest published evidence that an AND-composed
multi-verifier ensemble of FORMALLY DISTINCT graders provides robustness
against adversarial input crafting that a single (well-tuned) verifier
does not?** Cite specific papers with empirical attack/defense results.

**Q.B — What is the strongest published evidence that multi-verifier
ensembles FAIL to provide their advertised robustness in practice?**
Specifically, papers reporting joint null space failures, correlated
verifier errors, or ensemble robustness collapsing under capable adversaries.

**Q.C — How does the Sakana DGM open problem (self-improving agents
removing their own evaluators) connect to or differ from the broader
agent-safety / specification-gaming literature?** Is the
"externally-grounded multi-verifier ensemble" defense a recognized
research direction, or is it new?

**Q.D — What named architectures explicitly use AND-composed
multi-verifier ensembles for AI safety, and have been peer-reviewed?**
I want the comparator set Carnot must position against.

**Q.E — What is the consensus position (if any) on whether ensemble-
based defenses against reward hacking are a viable AGI-scale safety
mechanism, or are they fundamentally limited?**

### Output format requested

Please structure the report as:

1. **Executive summary** (3-5 paragraphs) covering the comparative
   landscape organized by architectural family
2. **Per-architecture sections** with two-paragraph intro + tabular
   per-work detail (primary reference, claim, evidence, lineage,
   open-source) + "Honest framing" subsection
3. **Comparative table** across all surveyed work: architecture |
   primary citation | benchmark category | peer-reviewed Y/N |
   open-source Y/N | scale | robustness claim
4. **Direct answers to Q.A through Q.E**
5. **Gaps and recommendations**: priority reading, architectures to
   compare against, novelty boundaries (what NOT to claim), researchers
   or labs to engage

### Format constraints

- Cite primary sources, not aggregator articles
- For arXiv preprints, include arXiv ID
- For peer-reviewed work, include venue + year
- Distinguish reproductions from original results
- For controversial claims, cite strongest evidence on each side
- For sparse axes, say so explicitly
- Distinguish ensembles of NEURAL classifiers (typical ML) from ensembles
  of FORMALLY DISTINCT verifiers (Carnot's case) — these are different problems

---

## Why this prompt now (decision-leverage)

DR-1 (energy-based LLM alternatives) confirmed that Carnot's strategic
positioning gap is "open-source externally-grounded EBM that solves
multimodal text collapse." The "externally-grounded" component is the
load-bearing novelty claim, but DR-1 didn't survey the verifier-ensemble
literature.

paper-v6's Sakana-defense argument depends on the multi-verifier ensemble
being:
- Genuinely novel (vs Constitutional AI, AI Safety via Debate, etc.)
- Theoretically principled (vs ad-hoc combinations)
- Empirically robust (vs joint-null-space failures)

Reviewers will ask "what's new vs constitutional AI or debate?" If
Carnot can't cite the comparative literature precisely, the central
novelty claim is vulnerable. This dive populates that defense.

The cost asymmetry is decisive: ~30 minutes of Deep Research now vs
the embarrassment of paper-v6 reviewers pointing out unrecognized
prior art on multi-verifier ensembles.
