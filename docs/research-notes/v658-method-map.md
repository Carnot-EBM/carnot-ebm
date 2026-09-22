# V658 method map

Date: 2026-09-22. Scope: advisory method accounting.
Paper results are not local Carnot results.

| Family | Primary source and section | Exact adaptation | Counterexample | Destination task |
|---|---|---|---|---|
| consistency | [arXiv:2606.08158v1 (2026-06-06)](https://arxiv.org/html/2606.08158), Section 2.1, constrained training objective | Constrain the two semantic option orders while retaining unconstrained and same-information controls. | Absent or mismatched source text is not label-preserving and cannot inherit the original label. | exp7521-consistency-energy, exp7522-source-evaluation |
| calibration | [arXiv:2609.11446v1 (2026-09-10)](https://arxiv.org/html/2609.11446v1), Sections 3-4 and Appendix B split protocol | Freeze probability calibration before selecting the accept, reject, or escalate cost policy. | Marginal calibration does not guarantee risk for each selected subgroup. | exp7517-source-protocol, exp7521-consistency-energy, exp7522-source-evaluation |
| online_proper_loss | [arXiv:2607.19689v1 (2026-07-22)](https://arxiv.org/html/2607.19689v1), Section 4 and Section 7.2 quadratic audit primitive | Compare chronological excess Brier loss with frozen, prevalence, and legally shuffled forecasts. | A count memory is not the Blackwell algorithm and delayed partial feedback inherits no theorem. | exp7523-count-memory, exp7524-count-online |
| thermodynamic_co_design | [Extropic Z1T (2026-09-04)](https://extropic.ai/writing/z1t), Porting Transformers, disaggregated inference, and hardware encoding | Preserve sparse operation shape and the complete FPGA-host service boundary. | Vendor estimates and a fast local kernel are not measured Carnot board or service speedups. | exp7528-service-boundary |

The primary method sections were read in a bounded low-concurrency pass.
Source access succeeded for these four rows. No benchmark number is a local result.
