import sys

refs = """
## 2026-05-09 Post-.124 Planning Sweep (Milestone 2026.05.125)

This sweep was run after milestone `.124` completed. Local outcomes: KANELÉ RTL logic synthesis and latency accounting landed, adaptive energy landscapes configured, EBM vs LLM task router smoke-tested. The failed tasks (gate blocked) point to a need for deeper integration of the optimizer with the local SOTA LLMs and proper Vivado linting integration.

### Energy-Based Reasoning Models (EBRMs) & Kona
- **Paper:** "Energy-Based Models for AI Reasoning: Beyond LLM Limitations" (Jan 2026).
- **What:** Introduces EBRMs which assign scalar energy scores to reasoning traces. Low energy indicates consistency with constraints. Kona uses continuous latent space reasoning with learned energy for local edits via gradient information.
- **Relevance to Carnot:** Extends the Phase 3 goal for verifiable multi-step reasoning. Carnot can adapt the EBRM pattern to evaluate the consistency of extracted constraint traces.
- **Concrete experiment hook:** Run an EBRM-style continuous latent trace scoring experiment to rank alternative reasoning traces extracted from local SOTA GGUF outputs.

### Πnet (Pi-net): Feasible-by-Design Neural Networks
- **Paper:** "Πnet (Pi-net): Feasible-by-Design Neural Networks" (Jan 2026).
- **What:** A neural architecture ensuring hard constraint satisfaction using a projection layer based on the Douglas-Rachford algorithm.
- **Relevance to Carnot:** Fits directly into Carnot's constraint satisfaction verification tier. If T-SKM / STATIC paths need reinforcement, Pi-net provides a differentiable projection layer alternative.
- **Concrete experiment hook:** Prototype a Pi-net style differentiable projection layer for a subset of the CCTU constraints and compare its projection accuracy vs T-SKM.

### Self-Modeling Generative Intelligence (SMGI)
- **Paper:** "SMGI (Self-Modeling Generative Intelligence)" (March 2026).
- **What:** Treats continual learning as a process of "certified updates" to ensure normative consistency and prevent catastrophic forgetting of constraints in explicit constraint systems.
- **Relevance to Carnot:** Directly addresses Carnot's continuous self-learning priority (FR-11 policy promotion). FR-11 memory growth needs certified updates to guarantee non-forgetting.
- **Concrete experiment hook:** Integrate SMGI "certified update" logic into the FR-11 policy promotion pipeline to gate memory growth on proven non-forgetting across a temporal constraint benchmark (like LTLZinc).

### Energy-Guided Decoding for LLMs
- **Paper:** "Energy-Guided Decoding for Object Hallucination Mitigation" (arXiv:2507.07731).
- **What:** Proposes dynamically selecting hidden states with the minimal energy score to reduce hallucination bias in vision-language and language models.
- **Relevance to Carnot:** Re-validates that explicit local energy scoring at the decoding step improves factual validity. 
- **Concrete experiment hook:** Incorporate energy-guided decoding using the local explicit Carnot energy to select among generation paths from the mandated SOTA GGUF models.

### ConsFormer for CSPs
- **Paper:** "ConsFormer: A Transformer-based Framework for CSPs" (ICML 2026).
- **What:** Self-supervised Transformer framework acting as a solution refiner for CSPs without requiring labeled feasible solutions.
- **Relevance to Carnot:** Could provide an alternative to the SATQuest/PySAT oracle or help refine hard constraint graphs before Ising sampling.
- **Concrete experiment hook:** Train a small ConsFormer-style refiner on FoVer CSPs to observe if it can pre-condition Ising samplers for faster energy convergence.
"""

with open("research-references.md", "a") as f:
    f.write(refs)
