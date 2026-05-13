# Mouth/Brain Separation Architecture Proposal

**Date**: 2026-05-13
**Author**: Gemini CLI
**Status**: Proposed

## Context
Recent Energy-Based Transformer (EBT) papers (arXiv:2507.02092) dictate separating the language generator ("mouth") from the energy verifier ("brain"). Currently, the Python `VerifyRepairPipeline` violates this separation by tightly coupling LLM loading and generation with the energy verification pipeline. 

## Current Coupling
- `VerifyRepairPipeline` internalizes HuggingFace `AutoModelForCausalLM` and `AutoTokenizer` instances.
- The `_generate()` method mixes LLM token generation directly into the constraint verification module.
- The `verify_and_repair()` loop hardcodes the generator-verifier interaction, making it impossible to scale the generator independently or swap it for an external API without rewriting the verification logic.
- The Rust layer (`carnot-constraints/src/pipeline.rs`) is already well-separated and does not handle generation, proving that the verifier can exist independently.

## Proposed Architecture
1. **The Brain (Verifier)**: 
   - `VerifyRepairPipeline` will be refactored into a pure `VerifierPipeline`. 
   - It will take a `question` and `response` and return a `VerificationResult`.
   - All references to `self._model`, `self._tokenizer`, and `self._generate()` will be removed.

2. **The Mouth (Generator)**:
   - A new `Generator` or `LanguageGenerator` class will encapsulate `AutoModelForCausalLM` loading and inference.
   - This class will expose a `generate(prompt: str) -> str` interface.

3. **The Orchestrator (Verify & Repair Loop)**:
   - A new orchestrator class or function (`RepairLoopOrchestrator`) will coordinate the Mouth and the Brain.
   - It will handle the iterative repair logic, calling `Generator.generate()` and passing the output to `VerifierPipeline.verify()`, formatted with natural language violation feedback.

## Benefits
- **Scalability**: The generator can be offloaded to a separate GPU cluster or an external API without affecting the verification pipeline.
- **Maintainability**: Aligns the Python architecture with the clean Rust implementation and modern EBT literature.
- **Testability**: The verifier can be unit-tested purely on text, and the generator can be tested independently of constraint logic.
