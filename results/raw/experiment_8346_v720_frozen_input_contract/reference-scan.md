## 2026-10-09 — V720 planning scan: finish measured learning and test spline deployment

This scan precedes V720 task design. It covers primary 2025–2026 papers and
the requested secondary sources. Rechecked papers are not new discoveries.
Paper results remain external evidence, not Carnot measurements.

### Methods to carry into the next plan

- **Sparse online updates:** Hoang, Gupta and Harris,
  [online KAN learning, 2602.02056v4](https://arxiv.org/html/2602.02056v4).
  Compact spline support permits sparse coefficient updates. Use this as the
  basis for the pending local learner. Check independent arithmetic, durable
  recovery and later decision utility separately. Locality alone does not
  establish lower error or lower complete-service cost.
- **Lookup-table deployment:** the same authors,
  [KANELÉ, 2512.12850v3](https://arxiv.org/abs/2512.12850v3), revised
  2026-06-16, maps bounded spline functions to discretized lookup tables.
  A useful new question is whether frozen Carnot sentence heads preserve
  typed decisions under table lookup and coefficient quantization. Measure
  table rebuild cost after updates. This is a CPU numerical adaptation;
  it does not reproduce the paper's hardware or training/pruning workflow.
- **Finite delayed feedback:** Ryabchenko, Attias and Roy,
  [capacity-constrained OCO, 2606.11711v1](https://arxiv.org/html/2606.11711v1).
  Finite tracking capacity can permanently lose delayed observations. Test
  occupancy, admission, feedback loss and restart invariants on constructed
  traces. A simple scheduler comparison does not inherit DW-FTRL regret bounds.
- **Verification granularity:** Kumar,
  [Verification Without Sufficiency, 2608.00585](https://arxiv.org/abs/2608.00585).
  Its multi-hop study finds that isolated chunk verification can discard
  useful evidence. Keep this limitation beside the sentence-head results.
  Do not infer whole-answer sufficiency from local scores. Decomposition
  needs its own future experiment and must not enter the frozen current study.
- **Representation controls:** [ARM–EBM equivalence, 2512.15605v4](https://arxiv.org/abs/2512.15605)
  motivates the existing probability-equivalent sigmoid control.
  [EBT, 2507.02092](https://arxiv.org/abs/2507.02092) motivates explicit
  compatibility energies. Neither makes a learned low energy a truth certificate.

### Broad scan and disposition

| Topic | Primary source | Decision |
|---|---|---|
| EBM reasoning and uncertainty | [Distributional EBMs, 2605.18871](https://arxiv.org/abs/2605.18871) | Separate learned quality from deterministic penalties; retain simple matched controls. |
| Neural constraint satisfaction / Ising optimization | [Neural LNS, 2603.20801](https://arxiv.org/abs/2603.20801) | Destroy/repair separation is promising for later search work; no new puzzle benchmark now. |
| Hallucination detection | [SURE-RAG, 2605.03534](https://huggingface.co/papers/2605.03534) | Controlled sufficiency and natural hallucination results differ; preserve independent human labels. |
| Energy-guided generation | [ETS, 2601.21484v3](https://arxiv.org/abs/2601.21484) | Rechecked ICML 2026 claim; defer steering until verifier utility and runtime qualify. |
| FPGA sampling | [Hybrid Ising decomposition, 2602.15985](https://arxiv.org/abs/2602.15985) | Include host decomposition and transfer costs; an Ising overlay is not a spline accelerator. |
| Constrained learning | [Safe-by-Design EBNNs, 2609.36942](https://arxiv.org/abs/2609.36942) | New September lead. Review formal assumptions before adoption; no claim that arbitrary extracted language constraints satisfy them. |

### Secondary-source checks and retrieval limits

- **OpenReview:** searched 2026 EBM and verification submissions. Search
  returned [EBT-related proceedings](https://openreview.net/pdf?id=ZBj3Qp1bYg)
  and [a small energy reward model submission](https://openreview.net/pdf?id=Kotvxxstmm).
  Both forum pages returned browser challenges. No unverified acceptance is asserted.
- **Semantic Scholar:** attempted citation endpoints for
  [EBT](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?fields=title,year,url&limit=10)
  and [ARM–EBM](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?fields=title,year,url&limit=10).
  Both returned retrieval errors. This scan cannot claim complete citation coverage.
- **Hugging Face Papers:** inspected SURE-RAG and the sufficiency paper above.
  [RT4CHART, 2603.27752](https://huggingface.co/papers/2603.27752) is a useful
  claim/span verification lead. Its reported annotation differences favor
  explicit label authority. It is not a replacement label source in this plan.
- **GitHub Trending:** checked monthly [Python](https://github.com/trending/python?since=monthly)
  and [Rust](https://github.com/trending/rust?since=monthly) pages. Returned
  snapshots were three weeks old. No current ranking or new trend is asserted.
  The primary [Torx repository](https://github.com/extropic-ai/torx) describes
  JAX stochastic circuits and directed factor graphs; simulation is not TSU access.
- **Extropic:** checked [Writing](https://extropic.ai/writing) and
  [Hardware](https://extropic.ai/hardware). The writing page exposed navigation
  only in this retrieval. Keep vendor roadmap claims separate from measured
  local hardware. No authenticated Carnot TSU access was found in project evidence.
- **Logical Intelligence:** [Kona 1.0](https://logicalintelligence.com/kona)
  still describes constraint-based reasoning. The retrieved page provides no
  reproducible equations or weights for an architecture reproduction.

These findings favor finishing qualified static and continuous decision tests.
Lookup-table fidelity and pending-feedback capacity are separate bounded studies.
No finding authorizes changing frozen evaluation targets or generator weights.
