# V676 method map — 2026-09-27

Exp7767 reviewed the adopted primary method sections after the V676 plan was
filed. This note records test ideas. It does not report Carnot science.

| Primary version and section read | Method and local transfer | Limit |
|---|---|---|
| [CCHD, arXiv:2606.08158v1](https://arxiv.org/html/2606.08158v1), June 6, 2026; §2.1–2.3 | The primary cross-entropy loss uses original document–claim pairs. One constraint bounds Jeffreys divergence between original and paraphrased predictions. Another bounds paraphrase cross-entropy against the label. Nonnegative Lagrange multipliers rise on violated constraints. Exp7771 compares analogous prediction consistency with matched ordinary augmentation and a small MLP. | Carnot's views preserve source bytes and use fixed windows. They are not CCHD back-translations or trained encoders. Equal predictions can be equally wrong. No published benefit transfers. |
| [Catalog faithfulness, arXiv:2608.10008v3](https://arxiv.org/html/2608.10008v3), August 22, 2026; §3.3–3.5 | The paper validates its catalog-membership instrument against 201 human judgments and measures its net bias. It elicits generic verbal confidence on a 0–100 scale, then checks ECE, Brier score and retained-set behavior. Exp7773 tests an explicit source-support event against generic confidence on the same source and answer bytes. | The paper tests catalog membership, not source entailment. Its corrected v3 matcher and calibration findings supersede v1; v1 search snippets are unsafe. The local test must independently validate its annotation instrument and count invalid responses. |
| [Verification Without Sufficiency, arXiv:2608.00585v1](https://arxiv.org/html/2608.00585v1), August 1, 2026; §III and §VII–XII | A per-chunk verifier can reject a premise needed jointly with another. Set-level and decomposition-conditioned checks restore the missing premise in the paper's multi-hop tests. Exp7768 keeps every source sentence and adds a two-premise fixture; Exp7772 measures source dependence. | Preserving bytes does not prove that a learned feature represents their joint meaning. Gold decomposition and answer slots are oracle controls, not deployable evidence. Missing support means unknown, not contradiction. |
| [Capacity-constrained delayed OCO, arXiv:2606.11711v1](https://arxiv.org/html/2606.11711v1), June 10, 2026; §2–4 | At most C pending rounds can be tracked. A scheduler selects observations under the cap and the base learner receives delayed weighted feedback. Exp7769 and Exp7774 record prediction before feedback, queue occupancy, dropped observations and bounded admissions. | Its regret bounds assume convex losses and the paper's tracking model. A discrete predicate bank inherits no theorem. A complete static bank is a required causal control. |

## Access and scope

All four primary HTML pages and the sections named above were accessible on
2026-09-27. The V676 [planning review](../../research-references.md) had
abstract-level access for some sources; this receipt adds method-level review.
OpenReview EBT forum access and Semantic Scholar citation lists remain limited
as recorded there. Those access gaps do not imply that no related work exists.
No local model was loaded for this review. The plan still defers KAN, PAL,
neural-to-Ising and hardware architecture claims until matched local evidence
exists. Natural annotation comparisons stay exposed-development evidence.
