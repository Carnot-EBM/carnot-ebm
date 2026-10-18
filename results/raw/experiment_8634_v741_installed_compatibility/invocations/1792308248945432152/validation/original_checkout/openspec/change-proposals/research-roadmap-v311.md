# Research Roadmap v311 Change Proposal

## Motivation
Milestone .310 faced several blocked paths:
- KV260 hardware execution was blocked due to SSH unreachability.
- GateMate hardware latency execution was blocked due to lack of a host communication interface (AXI/UIO).
- Crucial gate checks failed because upstream experiments (e.g., VGB repair ladder and FR-11 LogicVault) did not adhere to the strict `status="success"` artifact contract, emitting `status="complete"` or omitting it entirely.
- Artifacts for live SOTA generation experiments (3352, 3354) were missing.

## Changes Proposed
This milestone (.311) focuses on:
1. **Hardware Fixes**: Directly targeting the blockages on KV260 and GateMate platforms.
2. **Gate Resiliency**: Refactoring the output schemas of .310 experiments so that gate checks correctly register their completion, unblocking the Capstone and Continuous Learning Stress Test.
3. **Scaled Continuous Learning**: Extending the FR-11 Z3 Counterexample repair to 100 cases, and adding multi-agent synchronization for LogicVault.
4. **Retry Missing Tasks**: Re-running the missing live SOTA generation tasks to complete the empirical baseline.

## Alignment with Strategy
These changes directly align with the core architectural goals, particularly continuous self-learning (FR-11) and establishing the FPGA hardware path.

## Proposed Tasks
13 tasks are drafted, maintaining the balance between hardware smoke tests, live-generation verification, self-learning loops, and operational gating.