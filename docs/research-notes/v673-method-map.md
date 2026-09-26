# V673 method map

Date: 2026-09-26. This map records method inputs for Exp7726. The studies
below do not measure Carnot's V673 mechanisms or establish generalization.

| Method | Primary source and review depth | Local adaptation and limit |
|---|---|---|
| Set evidence sufficiency | [SURE-RAG](https://arxiv.org/abs/2605.03534), abstract and V673 reference review | Exp7728 compares sentence evidence-set coverage with one shared evidence location. Support, refutation and insufficient evidence remain distinct. Controlled multi-hop results do not transfer to natural hallucinations. |
| Input-side alignment | [HalluSpan/EviAlign](https://arxiv.org/abs/2608.15804), abstract and author method note | Exp7728 tests independent evidence locations for complete sentences. Carnot's fixed features do not reproduce the trained masked-token encoder or its span detector. |
| Draft-conditioned structure | [DCCD](https://arxiv.org/abs/2603.03305), abstract | Exp7729 compares draft-conditioned decisions with token-matched direct and two-pass controls. Valid JSON and semantic correctness are separate metrics. No 27B benefit is presumed. |
| Delayed feedback | [Capacity-Constrained Online Convex Optimization with Delayed Feedback](https://arxiv.org/abs/2606.11711), abstract | Exp7732–7733 bound pending feedback and record predictions before release. Discrete constraint admission inherits no convex-regret theorem. |

## Focused delta search

The arXiv search on 2026-09-26 found [Structure Snowballing](https://arxiv.org/abs/2604.06066),
already present in the V673 reference delta. It reinforces the syntax versus
semantics control for Exp7729; it does not add a task or establish a result for
the mandated model. Direct arXiv records for the four mapped methods were
reachable. Semantic Scholar citation reads for SURE-RAG and HalluSpan returned
access errors. Citation coverage and novelty remain incomplete. This was a
focused sequential delta check, not an exhaustive literature review.

V672's blocked contract and capstone verdicts remain unchanged. The preserved
V673 design SHA-256 before this task was
`79e99dead4a39dbc62d7819e3f5b7c0d782d0ec0be67da010a601b2c5b81de51`.
