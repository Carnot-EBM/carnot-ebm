## 2026-10-10 — V723 planning scan: label criteria and recoverable online services

This scan precedes V723 experiment design. External results are research leads,
not Carnot measurements. Rechecked sources are identified below.

### Findings to ingest

- **New: human label authority.** Valjakka et al.,
  [The Labeling Problem in Hallucination Detection Benchmarks,
  2610.08026v1](https://arxiv.org/html/2610.08026v1), October 6, 2026.
  The study separates reference faithfulness from factual correctness.
  Its 900 responses share 300 questions across three generators. One annotator
  labels the full panel; a second labels 300 responses. The independent unit is
  the question, not each response. Adopt explicit target definitions and paired,
  question-clustered error analysis. Do not treat reference disagreement as
  proof of falsehood or a released model-judge label as human truth.
- **New: actual release route.** The author's
  [LPHB repository](https://github.com/jova486/LPHB) includes
  `data/human_judge_release_master.csv`, its dictionary and read-only Level 1
  reproduction. This route requires no model inference. Label regeneration is
  disabled by default and must stay disabled. The
  [Dataverse V2 record](https://doi.org/10.7910/DVN/PCHISZ) lists CC BY-SA 4.0
  for annotations, metrics and documentation; answer text retains model terms.
  Browser access to Dataverse failed and its API returned HTTP 403. GitHub is
  an author-provided alternative, not permission to bypass access controls.
  Authenticate the commit, bytes and separate licenses before use. The release
  is an opportunity to study label validity; it is not a Qwen3.8 benchmark.
- **Rechecked: sparse learning.** Hoang, Gupta and Harris,
  [online KAN, 2602.02056v4](https://arxiv.org/abs/2602.02056), June 19 revision,
  uses local spline support for sparse on-chip updates. Adopt locality as the
  implementation basis for complete durable transaction measurements. Published
  FPGA timings do not include Carnot's host storage or establish local speedup.
- **Rechecked: delayed feedback.** Ryabchenko, Attias and Roy,
  [2606.11711](https://arxiv.org/abs/2606.11711), June 10, studies finite pending
  capacity and lost feedback. Preserve release order and explicit lost-feedback
  records. Do not borrow a convex regret bound for a different learner.
- **Rechecked: numerical deployment.**
  [KANELÉ, 2512.12850v3](https://arxiv.org/abs/2512.12850), June 16 revision,
  motivates lookup deployment. V722's separate dyadic-logit certificate is
  numerical evidence only. Keep the existing SciPy probability policy and its
  closed utility findings distinct. Further table expansion can wait until
  complete direct-service costs exist.

### Other primary topics checked

| Topic | Source | Decision and limit |
|---|---|---|
| EBM verification and reasoning | [EBT, 2507.02092](https://arxiv.org/abs/2507.02092); [ARM–EBM, 2512.15605v4](https://arxiv.org/abs/2512.15605) | Architectural context. Compatibility and likelihood are not factual correctness. No generator training. |
| Structured verification | [Distributional EBMs, 2605.18871](https://arxiv.org/abs/2605.18871) | Keep deterministic penalties separate from learned quality and model-identity shortcuts. No external-text reranker revival. |
| Neural constraints | [Certified Correctness, 2608.14569](https://arxiv.org/abs/2608.14569) | Position paper supports instance-level symbolic checking. Search metadata has inconsistent dates; no new October empirical claim. |
| Energy-guided decoding | [2507.07731](https://arxiv.org/abs/2507.07731); [draft-conditioned decoding, 2603.03305](https://arxiv.org/abs/2603.03305) | First is visual and outside scope. Defer local generation intervention until CUDA qualifies. |
| Ising in ML and hardware | [FPGA decomposition, 2602.15985](https://arxiv.org/abs/2602.15985); [parallel inertial Ising, 2604.17109](https://arxiv.org/abs/2604.17109) | Count preparation and transfers. Optimization success is not a Gibbs-distribution certificate. Neither provides a KV260 spline kernel. |
| Continual learning and KAN | Online KAN and delayed-capacity papers above | Use existing local updates and durable feedback; no new scheduler sweep. |

### Secondary channels and access limits

- **OpenReview:** checked the [ICLR 2026 EBT paper](https://openreview.net/pdf?id=ZBj3Qp1bYg)
  and [NRGPT comparison](https://openreview.net/pdf?id=B3Muyi2zgo). The EBT forum
  returned a browser challenge. Neither paper changes the local evidence gates.
- **Extropic:** checked the writing index, [Z1T](https://extropic.ai/writing/z1t/),
  [Torx and Thermalizers](https://extropic.ai/writing/from-one-to-one-billion),
  and [October research-agent update](https://extropic.ai/writing/baby-thermo-rsi/).
  The last describes a planned large-chip feedback loop in 2027. Vendor plans
  do not establish local TSU access. Retain execution-versus-benefit separation.
- **Semantic Scholar:** attempted graph citation queries for both
  `ARXIV:2507.02092` and `ARXIV:2512.15605`. Both failed retrieval. Site searches
  supplied no usable citation inventory. Citation coverage remains incomplete.
- **Hugging Face papers:** checked [OpenHalDet](https://huggingface.co/papers/2606.06959).
  The new label paper's page was unavailable. V722's missing independent label
  evidence remains valid for those old releases; the new author CSV is separate.
- **GitHub Trending:** checked [Python](https://github.com/trending/python?since=monthly)
  and [Rust](https://github.com/trending/rust?since=monthly). Returned snapshots
  were four weeks old. No claim about current ranking or new EBM repos follows.
  LPHB was found through its paper, not through Trending.
- **Logical Intelligence:** checked the [current site](https://logicalintelligence.com/)
  and [Kona page](https://logicalintelligence.com/kona). They provide product
  positioning, not reproducible implementation evidence for a new Carnot method.

