## 2026-10-08 — V718 planning scan: qualified local learning and bounded feedback memory

This scan precedes the V718 experiment design. It covers 2025–2026 work.
Existing entries remain authoritative about earlier scans. Rechecked papers are
not presented as new discoveries. Search snippets are leads, not measurements.

### Findings selected for design

- **Online spline learning.** Hoang, Gupta and Harris,
  [Ultrafast On-Chip Online Learning via Spline Locality in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2602.02056v4)
  (2026-06-19 revision; ICML 2026). The full text describes compact spline
  support and fixed-point training. Continue Carnot's unmeasured local-update
  question after qualifying the existing numerical evidence. Count coefficient
  touches and durable writes. Vendor or paper latency does not establish local
  FPGA speed. [Full text](https://arxiv.org/html/2602.02056v4).
- **Finite pending-feedback capacity.** Ryabchenko, Attias and Roy,
  [Capacity-Constrained Online Convex Optimization with Delayed Feedback](https://arxiv.org/abs/2606.11711)
  (2026-06-10). The paper separates a tracking scheduler from a delayed learner.
  Its guarantees require particular scheduling, weighting and loss assumptions.
  A useful bounded experiment tests queue capacity, permanent feedback loss,
  issue-before-release order and restart behavior. Compare unlimited tracking,
  deterministic admission and randomized admission on fixed constructed traces.
  This is a systems adaptation, not a reproduction of DW-FTRL or its regret
  bound. Keep it separate from the frozen natural-data learning hypothesis.
  [Full text](https://arxiv.org/html/2606.11711v1).
- **Evidence sufficiency.** Qiu, Han and Huang,
  [SURE-RAG](https://arxiv.org/abs/2605.03534v2) (2026-07-24 revision).
  Local claim/evidence relations can inform an answer-level selector. Its
  natural-hallucination boundary result cautions against transferring controlled
  sufficiency gains. Carnot must retain human response labels, same-input
  classifier controls and all missing source slots. Rechecked via
  [Hugging Face Papers](https://huggingface.co/papers/2605.03534).
- **Energy is a representation, not a correctness certificate.**
  [ARM–EBM equivalence](https://arxiv.org/abs/2512.15605v4)
  (2026-05-25 revision; [ICML proceedings](https://proceedings.mlr.press/v306/blondel26a.html))
  motivates an exactly probability-equivalent sigmoid control. It does not
  establish Carnot's verifier accuracy. [EBT](https://arxiv.org/abs/2507.02092)
  remains relevant background for explicit compatibility energies;
  [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e19a65fd53b6f9a88b354da98813465d-Abstract-Conference.html)
  independently identify its publication. No foundation-model training is
  justified by this small local study.

### Broad scan and disposition

| Area | Primary source | Planning disposition |
|---|---|---|
| Neural constraint satisfaction | [Construct-and-Refine, 2602.16012](https://arxiv.org/abs/2602.16012) | Feasibility and optimization are separate outcomes. Defer routing-domain expansion. |
| Reasoning verification | [Distributional EBMs, 2605.18871](https://arxiv.org/abs/2605.18871) | Keep uncertainty and abstention controls. Do not import reported benchmark gains. |
| Hallucination detection | [Diversion decoding, 2607.10476](https://arxiv.org/abs/2607.10476) | Watch list: needs new generation and does not resolve current qualification failures. |
| Energy-guided decoding | [ETS, 2601.21484](https://arxiv.org/abs/2601.21484) | Defer steering until a verifier earns decision benefit. |
| Ising / FPGA | [Dual-BRAM p-bit annealer, 2602.16143](https://arxiv.org/abs/2602.16143) | Memory traffic is relevant. Its ZC706 annealer does not implement Carnot's spline writes. |
| Ising decomposition | [Hybrid FPGA decomposition, 2602.15985](https://arxiv.org/abs/2602.15985) | Require compatible operations and full host/transfer costs before hardware claims. |
| Continual constraint systems | [Delayed capacity OCO, 2606.11711](https://arxiv.org/abs/2606.11711) | Add a bounded systems stress test beside the unchanged delayed-learning protocol. |

### Secondary sources checked and access limits

- **OpenReview:** searched ICLR 2026 EBM and reasoning submissions. The EBT
  [PDF endpoint](https://openreview.net/pdf/608231a168a72d241775e5d1d28a092f5532becb.pdf)
  returned a browser challenge. The intrinsic-optimizer
  [search result](https://openreview.net/pdf?id=UGB6JCl9lz) still says under review.
  Use the official EBT proceedings above. Do not infer other acceptance decisions.
- **Semantic Scholar:** attempted citations endpoints for
  [EBT](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?fields=title,year,url&limit=10)
  and [ARM–EBM](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?fields=title,year,url&limit=10).
  Both returned retrieval errors. Domain searches supplied no reliable citation
  list. Citation coverage remains incomplete; no citing-paper count is claimed.
- **Hugging Face Papers:** checked SURE-RAG and
  [SIRIN, 2608.00033](https://huggingface.co/papers/2608.00033).
  SIRIN's unified detector interface is a future comparator lead. It does not
  justify adding another detector before the pending sentence study executes.
- **GitHub Trending:** inspected monthly
  [Python](https://github.com/trending/python?since=monthly) and
  [Rust](https://github.com/trending/rust?since=monthly) pages. Returned snapshots
  were about three weeks old. No current EBM/KAN/constraint ranking is inferred.
  Separately inspected the primary [Torx repository](https://github.com/extropic-ai/torx),
  which describes JAX stochastic circuits and directed conditional samplers.
- **Extropic:** checked [Writing](https://extropic.ai/writing),
  [the summer announcement](https://extropic.ai/writing/from-one-to-one-billion)
  and [Hardware](https://extropic.ai/hardware). The writing search lists Torx,
  Thermalizers and Z1T updates. The hardware page lists Z1 Stick/Card early
  access in 2027. These are vendor roadmap claims. Carnot has no authenticated
  TSU access; neither Torx simulation nor a future device is current hardware.
- **Logical Intelligence:** checked [the main site](https://logicalintelligence.com/)
  and [Kona](https://logicalintelligence.com/kona). The retrieved pages expose
  product positioning, without enough equations, weights or reproducible
  architecture details for a new Kona reproduction experiment.

The immediate decision is to finish a qualified measurement of the existing
local mechanism. New delayed-capacity work has an independent bounded scope.
No paper authorizes changing frozen labels, relaxing scientific thresholds,
updating the mandated generator, or attributing simulated results to hardware.

