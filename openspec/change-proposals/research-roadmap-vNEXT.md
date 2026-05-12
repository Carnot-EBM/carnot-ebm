# Carnot Research Roadmap: Milestone 2026.05.149

**Milestone:** 2026.05.149
**Title:** Research-Deep: SOTA Runtime Recovery, Terminal Artifact Integration, Validator Ledgers, and EBM Foundations
**Status:** PROPOSED

## 1. Context & What Previous Milestones Proved

Milestone `.148` was designed to integrate SOTA GGUF models, low-cost telemetry, continuous self-learning ledgers, and hardware accounting. However, it encountered significant friction: the lack of available terminal artifacts for the SOTA cache/runtime preflight caused a cascade of gate blocks for downstream tasks. Furthermore, attempts to execute codex CLI scripts resulted in early exits.

Therefore, `.148` proved that:
1. **Terminal Artifact Dependency is Absolute:** Downstream tasks cannot safely run unless upstream tasks produce schema-complete terminal artifacts, even in failure modes.
2. **SOTA Runtime Readiness is the Core Blocker:** Until the tri-SOTA models (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF`) are reliably cached and preflighted, live evaluation is impossible.
3. **Telemetry and Ledgers Must Proceed Deterministically:** Spilled-energy diagnostics and residual drift ledgers must exist as pure advisory artifacts before they can gate repair.

Recent findings in EBM literature (e.g., *Energy-Based Transformers are Scalable Learners and Thinkers* [arXiv:2507.02092] and *Generalizable Reasoning through Compositional Energy Minimization* [arXiv:2510.20607]) confirm that Carnot's Phase 3 direction towards compositional constraint solving via energy minimization is aligned with the bleeding edge.

## 2. The 3 Biggest Gaps to the PRD Vision

1. **SOTA Runtime & Terminal Artifact Contracts:** The PRD mandates local-first execution. The failure of `.148` to unblock tri-SOTA inference means Carnot lacks the foundational execution substrate required to test continuous self-learning or verifiable output generation at the SOTA tier.
2. **Continuous Self-Learning & Non-Forgetting (FR-11):** The PRD requires Carnot to improve its own validation policies over time. The failure of the routing-without-forgetting promotion gate in `.148` leaves Carnot without an operational self-learning loop.
3. **Bridging the Gap to EBM Systematic Reasoning:** Current architecture uses LLMs to generate and verifiers to check. The PRD and latest research point to a future where constraints natively guide generation (e.g., EBTs and Compositional Energy Minimization). The gap between post-hoc checking and native constraint-guided generation remains large.

## 3. Milestone 2026.05.149 Phases

### Phase 1: Foundation Recovery & Terminal Telemetry (Tasks 1904-1907)
We must establish the execution baseline. We will write the `.149` activation contract, run a hardened preflight for the SOTA GGUF cache (ensuring a terminal artifact is written regardless of success), and establish a token-level telemetry adapter that records spilled energy and logprob data without modifying deterministic acceptance.
* **Key Arxiv Inclusion:** *Spilled Energy in LLMs* and *Energy-Based Transformers* concepts for diagnostic telemetry tracking.

### Phase 2: Continual Self-Learning & Validator Ledgers (Tasks 1908-1911)
We address the residual drift and self-learning gaps. By implementing a residual drift ledger over the compiled ROCE validator trees, we can differentiate outright contradiction from satisfiable drift. This phase will also re-attempt the FR-11 self-learning promotion gate, ensuring non-forgetting checks are honored.
* **Key Concept:** *Routing without Forgetting* applied to validator policies.

### Phase 3: Constrained Decoding & Compositional Interfaces (Tasks 1912-1913)
We compare baseline drafting, hard constrained decoding, and draft-conditioned repair to mitigate "structure snowballing." Additionally, we audit the interface of our existing validators to ensure they are compatible with non-autoregressive (e.g., Glauber/Diffusion) EBM sampling in the future.
* **Key Arxiv Inclusion:** *Structure Snowballing* risk guardrails and *Compositional Energy Minimization* preparation.

### Phase 4: Hardware Accounting & Independent Baseline Checks (Tasks 1914-1917)
We explicitly define hardware execution boundaries. We will implement the corrected Curie-Weiss THRML parity check (without claiming hardware execution), test GEM/ConsFormer graph preconditioning against classical baselines, and complete the p-bit/p-dit accounting. The milestone concludes with the `.149` integrated E2E tri-SOTA smoke and retro.

## 4. Hardware Requirements & Constraints
- **Required:** Multi-GPU Linux workstation (e.g., dual RTX 3090) or high-memory M-series Mac capable of loading the mandated GGUFs into RAM/VRAM.
- **Constraints:** NO hardware execution claims for TSU, KV260, or XTR-0 are permitted. All p-bit/p-dit and S2KAN work must remain as no-synthesis resource accounting or simulator-only parity (`hardware_execution_claim=false`).
- **Models:** `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF` (fallback to small models is forbidden for headline evaluation; task must yield if missing).
