# Google Deep Research — Energy-Based LLM Alternatives Survey (Synthesis)

**Status:** Response received 2026-05-03 ~21:35Z. Full PDF at `/tmp/Non-AutoregressiveLLMAlternativesSurvey.pdf` (815KB, ~25 pages, 38 cited works).
**Source prompt:** `energy-based-llm-alternatives-deep-research-prompt.md`
**Strategic role:** populate paper-v6 Related Work + sharpen novelty positioning

---

## TL;DR — six things that change the paper-v6 strategy

1. **EBT (arXiv:2507.02092, Gladstone et al. ICLR 2026)** outscales Transformer++ by 35% AND has public code (alexiglad/EBT). This solves "System 2 thinking via energy minimization without reward models." **Carnot must NOT claim novelty over this.**

2. **LLaDA (arXiv:2502.09992, Nie et al. ICLR 2026)** is the open-source gold standard — 8B masked-diffusion LM matching LLaMA3 8B at identical compute (10²³ FLOPs) on MMLU/GSM8K/BBH. **Carnot must benchmark against LLaDA explicitly.** Carnot must NOT claim novelty on "bidirectional generation solving reversal curse."

3. **Kona 1.0 (Logical Intelligence 2026)** ships closed-source EBRM commercially with Yann LeCun on the technical board. Solves 96.2% of Sudoku in 313ms without Python-execution cheats. **This is closed-source enterprise; Carnot's open-source positioning here is the differentiator.**

4. **Coconut (arXiv:2412.06769, Hao et al. 2024/2025)** is the open continuous-latent reasoning baseline — feeds last hidden state back as embedding, enabling latent BFS. **Carnot must compare to Coconut on training curriculum and compute overhead.**

5. **The consensus position is "COMPLEMENTARY not REPLACEMENT"** — EBM + AR LLM as semantic interface is the AGI vision. This is the consensus paper-v6 should align with, not fight.

6. **Carnot's positioning gap is "open-source externally-grounded EBM"** — Kona has external grounding but is closed; EBTs/NRGPT are open but lack external grounding. This is the strategic novelty axis.

---

## What we MUST cite (we missed before)

```
EBTs                    Gladstone et al. (2025/2026 ICLR)  arXiv:2507.02092
LLaDA                   Nie et al. (2025/2026 ICLR)        arXiv:2502.09992
Coconut                 Hao et al. (2024)                   arXiv:2412.06769
Kona 1.0                Logical Intelligence (2026)         logicalintelligence.com
ODAR                    Ma et al. (2026)                    arXiv:2602.23681
Energy Transformer      Hoover et al. (2023 NeurIPS)        (predecessor to EBTs)
Omni-Diffusion          Li et al. (2026)                    arXiv:2603.06577
Modern Hopfield Nets    Ramsauer et al. (2020)              (foundational lineage)
JEPA                    LeCun lineage                       (must acknowledge for "continuous reasoning")
```

---

## The 4 "Honest Framings" by architecture family (verbatim distillation)

### Family 1: EBTs (Energy-Based Transformers)

**Claim:** System 2 thinking emerges from unsupervised energy minimization. Outscales Transformer++ by 35% on data/batch/params/FLOPs.

**Honest limit:** "EBTs encounter severe friction when modeling highly multimodal discrete distributions, such as natural language syntax and semantics. Researchers attempting to reproduce bidirectional EBTs for language modeling frequently report severe mode collapse... Consequently, while EBTs excel in continuous, spatial, or highly verifiable domains... standard autoregressive LLMs remain overwhelmingly superior for unconstrained, creative text generation."

**Carnot relevance:** EBTs solve "System 2 from unsupervised learning" — this is one of Carnot's positioning claims. Cannot claim novelty here. But Carnot's **externally-grounded** approach is differentiated from EBT's purely internal energy bound.

### Family 2: Score-Based / Diffusion Language Models

**Claim:** LLaDA matches LLaMA3 8B at identical compute, natively solves reversal curse.

**Honest limit:** "While theoretically parallelizable across GPU clusters during generation, the practical wall-clock latency for discrete diffusion on standard consumer or enterprise hardware still lags notably behind highly-engineered causal LLMs... the literature currently lacks evidence that text diffusion models scale gracefully and predictably to the 100B–400B parameter frontier."

**Carnot relevance:** Carnot is NOT a diffusion model. Position vs LLaDA: "we are an externally-grounded EBM, not a masked diffusion model. The reversal-curse benefit is not Carnot's primary axis."

### Family 3: Energy-Based GPT-Class (NRGPT)

**Claim:** Theoretically unifies GPT mechanics with energy-based framework via preconditioned gradient descent.

**Honest limit:** "The authors are remarkably honest about the model's empirical limitations... 'doesn't necessarily lead to the best performing models' for generalized tasks... NRGPT exhibits a phenomenon the authors define as 'asymptotic stability' that makes it initially resistant to standard overfitting, [but] is documented to suffer from catastrophic overfitting during very long training runs, revealing a severe scaling instability that standard autoregressive GPTs inherently avoid."

**Carnot relevance:** Carnot's Phase-4 already integrates NRGPT. **Q10's interpretation (cascaded multi-agent inference, sequential thermalization) IS the right framing.** This Deep Research confirms Q10's analysis. paper-v6 plays this correctly by distinguishing Regime 1 (monolithic, Carnot-style) from Regime 2 (cascaded, NRGPT-style).

### Family 4: Continuous-Latent Reasoning (Coconut, Kona)

**Claim:** Reasoning in continuous space avoids token-commitment errors. Coconut: 5% MathQA gain, latent BFS. Kona: 96.2% Sudoku in 313ms, "first energy-based reasoning model" commercially deployed.

**Honest limit:** Both are highly specialized. Kona is closed-source, mission-critical-only. Coconut requires complex multi-stage curriculum.

**Carnot relevance:** **This is the closest cousin to Carnot's Phase-3 architecture (DBAE-EBM with bounded latent z).** Carnot's positioning vs Coconut: open-source + multi-verifier external grounding. Carnot's positioning vs Kona: open-source + general-purpose (not mission-critical-only).

### Family 5: Hybrid / Active Inference (ODAR, SeedIQ)

**Claim:** Active inference for routing/orchestration of existing LLMs. SeedIQ: 100% ARC-AGI-3 (proprietary, refused code release sacrificing prize money).

**Honest limit:** These are CONTROL SYSTEMS WRAPPED AROUND LLMs, not LLM replacements. ODAR routes between Fast/Slow LLMs via free energy. SeedIQ is closed.

**Carnot relevance:** Carnot's Phase-4 active-inference work (exp1156 sampler + exp1165 ARC pilot) is in this family. **The ODAR paper (arXiv:2602.23681) is a missing citation.** SeedIQ's refusal to open-source aligns with our `documented_fallback` framing in the paper.

---

## Direct answers to Q.A through Q.E (load-bearing for paper-v6 framing)

### Q.A — Strongest evidence of MATCH/EXCEED on a task class

- **General-purpose LLM:** LLaDA matches LLaMA3 8B
- **Strict logic/constraint planning:** Kona 1.0 (96.2% Sudoku in 313ms, no Python-execution cheats)
- **Open-source latent reasoning:** Coconut (+5% MathQA, latent BFS surpasses CoT on planning)

### Q.B — Strongest evidence of FAIL TO SCALE in ways AR doesn't

- **Multimodal distribution collapse** (Family 1): bidirectional EBTs default to repetitive low-entropy outputs at language modeling
- **Training instability** (Family 3): NRGPT documented catastrophic overfitting on long training runs
- **Latency bottlenecks** (Family 2): LLaDA's iterative forward passes vs AR's KV-cache optimization

### Q.C — Most credible continuous-latent reasoning (2025-2026)

- Coconut (open-source, peer-reviewed implementations)
- Kona 1.0 (closed-source commercial; Yann LeCun on board)
- (SeedIQ excluded — code refused)

### Q.D — Peer-reviewed (ICLR 2026 main track) non-AR alternatives

- LLaDA
- EBTs
- NRGPT

All three explicitly position as non-AR alternatives, all peer-reviewed at the highest tier.

### Q.E — Consensus position (industry + academic)

> "energy-based alternatives are **complementary, not replacements**, for broad language generation, but are **strictly superior** for formal logic, verification, and execution governance. Standard autoregressive models excel as intuitive 'System 1' communicators, semantic interfaces, and broad knowledge retrievers... EBMs, diffusion models, and continuous-latent systems, however, excel at 'System 2' global constraint satisfaction where logical failures are fatal. The consensus vision for AGI and production enterprise systems is a multi-modal ecosystem: an EBM or continuous-latent model executes internal, multi-path reasoning to solve a complex, constrained problem, and an autoregressive language model acts as the semantic interface to translate that verified latent solution into human-readable text."

**Carnot must align paper-v6 with this consensus.** Position Carnot as the open-source externally-grounded "System 2" component of the multi-modal ecosystem, NOT as a wholesale replacement for AR LLMs.

---

## Novelty Boundaries (what NOT to claim — paper-v6 critical)

Per Deep Research's explicit guidance:

```
DO NOT claim novelty over:
  - "energy minimization for System 2 thinking without reward models"
    → EBT (Gladstone 2025/2026 ICLR) owns this. Comprehensively solved.
  
  - "bidirectional generation solving the reversal curse"
    → LLaDA (Nie 2025) decisively claimed and proved this territory.
  
  - "reasoning in continuous space rather than discrete tokens"
    → Must heavily acknowledge prior art:
       - LeCun's JEPA
       - Coconut latent embeddings
       - Kona's commercial deployment

MUST claim novelty (the strategic gap):
  - "open-source EXTERNALLY-GROUNDED EBM"
    → Kona has external grounding but is closed-source.
    → EBTs and NRGPT are open-source but lack external grounding.
    → Carnot fills this gap.
  
  - "multi-verifier ensemble defending against in-situ training reward hacking"
    → No published work on this exact intersection.
    → Carnot's Phase-5 derisking work (Q9 cross-validated) is novel here.
```

---

## Architectures Carnot MUST benchmark against in paper-v6

Per Deep Research:

1. **NRGPT** — to demonstrate external grounding differs from purely-theoretical internal integration into causal GPT masks
2. **LLaDA** — gold standard for open-source non-autoregressive text generation
3. **Coconut** — to prove training curriculum efficiency / computational overhead claims

If Carnot doesn't benchmark against these three, reviewers will flag the omission. Each is publicly cited, has open-source code, and has well-documented benchmarks.

---

## Carnot's strategic positioning (Deep Research's recommendation)

> "To maximize the impact and adoption of the proposed framework, the project should position itself squarely in the **'hybrid execution and verifiable grounding' gap**. While models like Kona achieve incredible verifiable grounding for formal logic, they are locked behind closed-source commercial silos tailored for enterprise infrastructure. Conversely, while EBTs and NRGPT are fully open source, they lack robust external grounding mechanisms, remaining isolated within their unsupervised, internal energy bounds. By explicitly defining the framework as an **externally-grounded EBM that solves the multimodal text collapse problem currently plaguing bidirectional EBTs**, the project will directly address the most significant, unsolved vulnerability in the current open-source non-autoregressive landscape."

**This is the paper-v6 thesis sentence.** Carnot = open-source + externally-grounded + solves multimodal text collapse.

---

## Q10 cross-validation: confirmed by independent literature review

Q10's interpretation that NRGPT exhibits "cascaded multi-agent inference / sequential thermalization" is independently confirmed by Deep Research:

> "[NRGPT] performs causal language modeling by minimizing a *per-token* energy rather than a sequence-level energy, introducing learnable inference rate matrices..."

Plus the explicit Q10 finding that NRGPT authors trade monotonicity for AUROC:

> "the authors explicitly concede that casting inference as energy exploration 'doesn't necessarily lead to the best performing models'..."

This validates the paper-v6 dual-regime framing (Regime 1 monolithic, Regime 2 cascaded multi-agent). **Q10's analysis was correct.**

---

## Recommended next steps

1. ✅ **Save this synthesis** (done — this file).
2. **Update paper-v6 draft** to:
   - Add EBT, LLaDA, Coconut, Kona, ODAR to bibliography
   - Insert Related Work section structured around the 5 architecture families
   - Apply the novelty-boundary discipline from above
   - Adopt the "externally-grounded EBM solving multimodal text collapse" thesis sentence
3. **Update ISSUE-16** in paper integrity audit — bibliography stub validation can now reference real papers
4. **Update Phase-5 derisking acceptance criteria** to include benchmark comparisons against NRGPT, LLaDA, Coconut at intermediate-scale (.96/.97 exp_NEXT_E)
5. **File new known-issues entry** for paper-v6 Related Work overhaul as .94/.95 mandatory

---

## Cross-validation status

```
Q7-Q10  Deep Think rounds                    (all clean)
DR-1   Energy-based LLM alternatives survey  (this — clean, 38 cited works,
                                               6 strategic findings, alignment
                                               with Q10 confirmed)

Pattern: Deep Research provides comprehensive landscape that Deep Think
focused architectural review cannot. They're complementary tools.
DR-1 surfaced 5+ papers we hadn't seen and would have missed in paper-v6.
```
