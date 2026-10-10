## 2026-10-09 — V721 planning scan: decision-preserving online table deployment

Recorded before writing the V721 experiments. This scan rechecks 2025–2026
primary work and all requested secondary sources. Revalidation is not a new
discovery. External results do not count as Carnot measurements.

### Methods selected for bounded follow-up

- **KANELÉ**, Hoang, Gupta and Harris, December 2025, revised June 16,
  2026: [primary v3](https://arxiv.org/html/2512.12850v3),
  [author code](https://github.com/Duchstf/KANELE). The paper maps bounded
  spline functions to quantized lookup logic. Carnot's Exp8352 already tests
  numerical tables and sparse refresh. The next question is different:
  preserve the direct policy's decisions at thresholds and publish updated
  coefficients, tables and error bounds as one durable version. Conservative
  interval fallback is our proposed adaptation, not a theorem from this paper.
  The paper's FPGA toolflow and speedups do not establish local fabric support.
- **Online KAN locality**, the same authors, February 2026, revised June 19:
  [primary v4](https://arxiv.org/html/2602.02056v4). Compact support permits
  sparse coefficient writes. Combine the existing update kernel with table
  invalidation and crash recovery. Measure the entire transaction, including
  bounds, persistence and fallback. Do not rerun sparse arithmetic alone or
  infer natural learning benefit from exact numerical agreement.
- **Delayed tracking capacity**, Ryabchenko, Attias and Roy, June 10, 2026:
  [primary](https://arxiv.org/abs/2606.11711). Pending observations can be
  permanently lost under finite capacity. Exp8349 already qualifies the
  constructed scheduler study. Retain its issue/release and loss semantics
  when testing durable deployment; no new scheduler sweep is justified here.
  Our learner does not inherit the paper's weighted-FTRL regret guarantees.
- **OpenHalDet**, June 2026:
  [primary](https://arxiv.org/abs/2606.06959), discovered through
  [Hugging Face Papers](https://huggingface.co/papers/2606.06959).
  Its standardized task and detector-access comparisons suggest a future
  external evaluation panel. First audit licenses, source overlap, labels and
  local inference cost. Do not import a new corpus into the frozen V717 study.
- **Certified neural constraint reasoning**, Kong, Zhang and Liu:
  [primary](https://arxiv.org/abs/2608.14569). The position paper separates
  neural proposals from symbolic certification. Apply that distinction to
  deployment: matching the direct numerical policy is an engineering claim;
  it does not certify the answer's semantic truth. Search metadata gave a
  conflicting date, so this scan cites the identifier without a precise day.

### Requested topic coverage

| Topic | Primary source checked | Disposition |
|---|---|---|
| EBM verification/reasoning | [EBT, 2507.02092](https://arxiv.org/abs/2507.02092); [distributional EBM, 2605.18871](https://arxiv.org/abs/2605.18871) | Keep learned scores distinct from hard constraint checks; no foundation-model training. |
| Neural constraint satisfaction | Certified reasoning above | Numerical fidelity does not establish faithful language extraction. |
| Ising applications and FPGA sampling | [Hybrid decomposition, 2602.15985](https://arxiv.org/abs/2602.15985) | Account for preprocessing, transfer and unsupported operations. |
| Hallucination detection | OpenHalDet above | External-evaluation lead; defer collection until qualification and runtime permit it. |
| KAN | KANELÉ and online KAN above | Test threshold fallback, atomic refresh and native deployment costs. |
| Energy-guided generation | [ETS, 2601.21484v3](https://arxiv.org/abs/2601.21484); [object-hallucination decoding, 2507.07731](https://arxiv.org/abs/2507.07731) | Defer steering until decision utility and live runtime qualify; the latter is vision-language work. |
| Continual/online learning | Online KAN and delayed-capacity OCO above | Preserve causal delayed updates while testing durable serving behavior. |
| ARM–EBM representation | [2512.15605](https://arxiv.org/abs/2512.15605) | Preserve the exactly equivalent sigmoid control; energy notation alone supplies no advantage. |

### Secondary sources and retrieval limits

- **OpenReview:** searched 2026 EBM/reasoning submissions. The
  [EBT forum](https://openreview.net/forum?id=ZBj3Qp1bYg) returned a browser
  challenge; indexed proceedings PDFs were visible. No new acceptance or
  independent reproduction is asserted from those snippets.
- **Semantic Scholar:** attempted citation endpoints for
  [EBT](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?fields=title,year,url&limit=10)
  and [ARM–EBM](https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?fields=title,year,url&limit=10).
  Both browser retrievals failed. Citation coverage is incomplete, not zero.
- **Hugging Face Papers:** checked verification results and followed
  OpenHalDet to the primary paper. Feed summaries are discovery aids.
- **GitHub Trending:** checked weekly [Python](https://github.com/trending/python?since=weekly)
  and [Rust](https://github.com/trending/rust?since=weekly). Returned snapshots
  were three and four weeks old. No current trending-repository claim follows.
  The [Torx author repository](https://github.com/extropic-ai/torx) remains
  a JAX stochastic-programming lead, not evidence of access to a TSU.
- **Extropic:** checked [Writing](https://extropic.ai/writing) and the
  September 4 [Z1T article](https://extropic.ai/writing/z1t/). Its efficiency
  model excludes inter-device data movement and final vocabulary logits in
  the main comparison; its latency estimate also excludes final logits.
  This strengthens the case for full transaction accounting. It supplies
  neither a local measurement nor an acquisition requirement for V721.
- **Logical Intelligence:** checked [Kona](https://logicalintelligence.com/kona).
  The retrieved architecture description provides no reproducible equations,
  weights or training recipe that changes this bounded plan.

The research decision is to reuse qualified evidence, close the two exact
audit coverage failures, and test deployment fidelity under continuous updates.
No source authorizes changing exposed evaluation targets or generator weights.

### V721 activation-refusal recheck — 2026-10-09

The bounded recheck preserves the preceding research design. Searches covered
EBM verification, neural constraints, Ising hardware, KANs, hallucination control
and online learning. The [online KAN paper](https://arxiv.org/abs/2602.02056)
and [delayed-capacity paper](https://arxiv.org/abs/2606.11711) still motivate the
existing local-update and feedback contracts. Revisited
[Token-Guard](https://arxiv.org/abs/2601.21969), already in this ledger, remains
a future decoding lead; this correction adds no generation experiment.

Secondary checks revisited OpenReview EBT search results, Hugging Face's
[OpenHalDet page](https://huggingface.co/papers/2606.06959),
[Extropic Writing](https://extropic.ai/writing), its
[Z1T article](https://extropic.ai/writing/z1t/),
[Kona](https://logicalintelligence.com/kona), and
[GitHub Trending](https://github.com/trending/python?since=weekly).
The GitHub snapshot was three weeks old. Both Semantic Scholar citation
endpoints listed above again failed retrieval. No new method adoption,
hardware-access claim or complete citation survey follows. The only task
change is the missing Exp8367 failure-lineage declaration.
