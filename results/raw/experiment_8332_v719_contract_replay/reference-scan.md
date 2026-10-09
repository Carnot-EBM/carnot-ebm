## V719 planning research — 2026-10-09, recorded before experiment design

The scan rechecked primary 2025–2026 papers and the requested secondary sites.
Previously catalogued work is identified as revalidation. A retrieved abstract
supports a method lead, not a claim that Carnot reproduced the paper.

### Methods worth testing now

- **Online KAN (revalidated):** Hoang, Gupta and Harris,
  [Ultrafast On-Chip Online Learning via Spline Locality in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2602.02056v4),
  February 2026, revised June 19, ICML 2026. Local spline support motivates
  sparse coefficient updates and explicit memory-write accounting. Test exact
  sparse/dense agreement, delayed natural feedback and complete transaction
  cost. Published fixed-point FPGA performance is not a Carnot board result.
- **Finite pending feedback (revalidated):** Ryabchenko, Attias and Roy,
  [Capacity-Constrained Online Convex Optimization with Delayed Feedback](https://arxiv.org/abs/2606.11711),
  June 10, 2026. Finite tracking capacity can discard feedback permanently.
  The paper uses randomized tracking and weighted delayed learning. This
  motivates a bounded admission-policy comparison with observable delay
  expirations, logged random priorities and explicit lost-feedback rows.
  A small fixed-schedule experiment does not inherit the paper's regret bound.
- **Evidence sufficiency (revalidated):** Qiu, Han and Huang,
  [SURE-RAG](https://arxiv.org/abs/2605.03534v2), May 2026, revised July 24.
  Joint evidence coverage and conflict matter beyond passage relevance.
  Preserve same-information controls and the natural-data boundary when
  testing the existing local sentence calibration protocol.
- **Additional boundary lead:** Kumar,
  [Verification Without Sufficiency](https://arxiv.org/abs/2608.00585), August 2026,
  surfaced through [Hugging Face Papers](https://huggingface.co/papers/2608.00585).
  A chunk can be insufficient alone while useful in a multi-hop proof. Use this
  as a claim limit: poor local sentence utility cannot retire all joint
  constraint reasoning. Defer a new decomposition experiment until the existing
  local study executes; do not change its registered targets after exposure.
- **Toolkit lead:** [SIRIN](https://arxiv.org/abs/2608.00033), 2026,
  and its [author repository](https://github.com/sb-ai-lab/SIRIN), discovered via
  [Hugging Face](https://huggingface.co/papers/2608.00033). Its common interface
  distinguishes representation, uncertainty and judge-based detection, including
  memory faithfulness. Consider later interoperability; no dependency or new
  model is needed for the current cached-data question.

### Topic coverage and deferred branches

| Topic | Primary source checked | Planning consequence |
|---|---|---|
| EBM reasoning | [EBT, July 2025](https://arxiv.org/abs/2507.02092); [ARM–EBM, December 2025, v4 May 2026](https://arxiv.org/abs/2512.15605v4) | Preserve the exactly equivalent sigmoid control. An energy parameterization alone does not prove correctness. |
| EBM verification | [Distributional EBMs, May 2026](https://arxiv.org/abs/2605.18871) | Compare learned quality with matched simple controls; preserve generator-shortcut and oracle boundaries. The retired generic text-ranker family stays closed. |
| Neural constraints | [HardNet++, April 2026](https://arxiv.org/abs/2604.19669); [CAffNet, May 2026](https://arxiv.org/abs/2605.24437) | Constraint adherence is separate from faithful extraction. No new projection stack this milestone. |
| Ising/FPGA sampling | [Dual-BRAM p-bit annealer, February 2026](https://arxiv.org/abs/2602.16143); [FPGA/ASIC decomposition, February 2026](https://arxiv.org/abs/2602.15985) | Measure memory, host work and transfer. Annealer throughput does not measure a spline update. |
| Hallucination mitigation | SURE-RAG, Verification Without Sufficiency and SIRIN above | Separate useful decisions, missing evidence and calibration from mechanical extraction. |
| KAN and continual learning | Online KAN and capacity-constrained OCO above | Finish local adaptation and test finite pending-state costs. |
| Energy-guided decoding | [Energy-Guided Decoding for Object Hallucination Mitigation, July 2025](https://arxiv.org/abs/2507.07731); [Verifier-Guided Decoding, July 2026](https://arxiv.org/abs/2607.27823) | These are vision-language methods. Do not infer text-verifier benefit or add a multimodal branch. |

### Secondary-source checks and retrieval limits

- **OpenReview:** searched ICLR 2026 EBM submissions and inspected the indexed
  [EBT proceedings](https://openreview.net/pdf?id=ZBj3Qp1bYg). A second energy-model
  [paper URL](https://openreview.net/pdf?id=B3Muyi2zgo) returned a browser challenge.
  No uninspected submission is treated as a verified new method.
- **Semantic Scholar:** searched both anchor IDs and attempted Graph API
  citation endpoints for ARXIV:2507.02092 and ARXIV:2512.15605. Both returned
  browser-tool errors. The citing-paper inventory remains incomplete; this
  does not imply zero citations. Anchor metadata was checked on arXiv directly.
- **Extropic:** [Writing](https://extropic.ai/writing) returned a sparse index;
  the dated [Z1T article](https://extropic.ai/writing/z1t/) was readable.
  Its September 4, 2026 co-design study maps sparse operations across Z1 and
  FPGA companions and reports estimates. Account for incompatible operations,
  readout and transfer before considering whole-service acceleration. This
  supplies no local TSU access or measured Carnot efficiency.
- **GitHub Trending:** weekly [Python](https://github.com/trending/python?since=weekly)
  and [Rust](https://github.com/trending/rust?since=weekly) returned snapshots
  crawled three weeks earlier. No current trend rank or newly trending
  EBM/constraint/KAN repository is asserted.
- **Logical Intelligence:** [Kona](https://logicalintelligence.com/kona) was
  checked for architecture updates. The retrieved vendor page supplies no
  inspected reproducible training recipe that changes this milestone.

The selected sources favor finishing a bounded local decision and learning
study. New literature does not justify renaming an unexecuted study, changing
its exposed-data status, relaxing validation, or claiming hardware speedups.
