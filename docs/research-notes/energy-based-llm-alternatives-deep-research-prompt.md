# Google Deep Research — Energy-Based Alternatives to Autoregressive LLMs

**Status:** PROMPT — used by operator 2026-05-03 ~21:30Z, response received as PDF
**Strategic role:** populated paper-v6 Related Work + sharpened novelty positioning
**Output:** `energy-based-llm-alternatives-deep-research-source.pdf` (815KB, 38 cited works)
**Synthesis:** `energy-based-llm-alternatives-deep-research-results.md`

---

## The Deep Research request

I am surveying the literature for a research project building an open-source
energy-based-model framework that positions itself as a non-autoregressive,
externally-grounded alternative to autoregressive transformer LLMs. I need
a comprehensive survey of the comparative landscape so the project's
"Related Work" section is honest and thorough about what's been done.

Please produce a structured research report on **energy-based, score-based,
diffusion-based, and otherwise non-autoregressive alternatives to standard
autoregressive transformer LLMs**, focused on work published 2023-2026.

### Survey scope (in priority order)

**1. Energy-Based Transformers (EBT-class)** — architectures replacing softmax/cross-entropy with energy minimization. Modern Hopfield network lineage. EBT/EBM-Transformer recent work.

**2. Score-Based / Diffusion Language Models** — discrete or continuous diffusion for text generation. Diffusion of Thought, absorbing-state diffusion, latent diffusion for text.

**3. Energy-Based GPT-Class Alternatives** — retain GPT scaffolding but replace cross-entropy with energy mechanics. NRGPT (arXiv:2512.16762) lineage.

**4. Continuous-Latent Reasoning Substrates** — cognition-in-latent-space. Logical Intelligence Kona, Themesis SeedIQ AΩ FoB HMC, Coconut, latent program induction, active inference applied to LLM substrates.

**5. Hybrid / Adjacent** — Mixture-of-Experts with energy criteria, reasoning models (o1/o3 class), recurrent transformer alternatives (Mamba/RWKV/RetNet), test-time-compute scaling.

### What I already cite (deprioritize unless recent developments)

NRGPT (arXiv:2512.16762), Diffusion of Thought, Friston canonical work, Modern Hopfield Networks (Ramsauer 2020), Du & Mordatch 2019, BEAVER-lite (arXiv:2512.05439), Apple SSD.

### What I want to learn

For each architecture: (1) what it claims, (2) what it demonstrates (empirical scale), (3) primary reference with arXiv ID, (4) lineage, (5) open-source status, (6) honest comparison framing to autoregressive LLMs.

Particularly: recent work I haven't seen, negative or ambiguous results, theoretical critiques, industrial adoption signals.

### Specific questions

**Q.A** — What is the strongest published evidence that an energy-based or non-autoregressive alternative MATCHES OR EXCEEDS autoregressive LLM performance on a task class?

**Q.B** — What is the strongest published evidence that energy-based / score-based / diffusion language models FAIL TO SCALE in ways autoregressive transformers do not?

**Q.C** — What is the most credible recent (2025-2026) implementation of "continuous-latent reasoning" reporting non-trivial performance?

**Q.D** — What named architectures explicitly position themselves as "alternatives to autoregressive LLMs" and have been peer-reviewed (not just arXiv preprints)?

**Q.E** — What is the consensus position (if any) on whether energy-based alternatives can replace, complement, or are strictly inferior to autoregressive transformers at scale?

### Output format

1. Executive summary (3-5 paragraphs): comparative landscape by architectural family
2. Per-architecture sections with two-paragraph intro + tabular per-work detail (primary reference, claim, evidence, lineage, open-source) + "Honest Framing" subsection
3. Comparative table: architecture | citation | benchmark | peer-reviewed | open-source | scale | parity claim
4. Direct answers to Q.A through Q.E
5. Gaps and recommendations: priority reading, architectures to compare against, novelty boundaries (what NOT to claim), researchers to reach out to

### Format constraints

- Cite primary sources, not aggregator articles
- arXiv IDs for preprints, venue+year for peer-reviewed
- Distinguish reproductions from original results
- For sparse axes, say so explicitly
- For controversial claims, cite strongest evidence on each side
