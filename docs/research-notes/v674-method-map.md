# V674 method map — 2026-09-27

This map records how the reviewed methods change a falsifiable local test.
It does not qualify a learned decision or a hardware benefit.

| Primary source and review depth | V674 test | Local adaptation and limit |
|---|---|---|
| [HalluSpan/EviAlign](https://arxiv.org/html/2608.15804v1), training, evaluation and limitations read | Exp7740 binds complete sentence targets; Exp7743 compares local and response-only supervision; Exp7744 tests calibrated cost and source-erased control | The paper trains a masked-span encoder using output labels and no gold input alignments. Carnot uses its existing finite features and sentence targets, not that encoder. High-similarity conflict errors and per-token re-encoding cost are explicit failure risks. Exposed RAGTruth remains development only. |
| [EBT](https://arxiv.org/abs/2507.02092) and [ARM–EBM](https://arxiv.org/abs/2512.15605), abstracts reviewed | Exp7741 tests finite normalization and typed decision execution | Energy ranking does not certify truth. Generator weights stay fixed. |
| [Capacity-constrained online convex optimization with delayed feedback](https://arxiv.org/abs/2606.11711), abstract and capacity assumption reviewed | Exp7742 and Exp7746 track pending credits, frozen predictions and retention under delayed labels | Carnot admits discrete advisory predicates. The paper's convex regret bound does not transfer. A complete static dictionary and frozen bank are controls. |
| [DCCD](https://arxiv.org/abs/2603.03305) and [structural/semantic gap](https://arxiv.org/abs/2609.23742), abstracts reviewed | Exp7745 compares direct and localized Qwen decisions at equal token budget | V673's draft null remains. Parsing, quotations and semantic decisions get separate metrics. |
| [KAN forgetting](https://arxiv.org/abs/2511.12828) and [KAN-CL](https://arxiv.org/abs/2605.12306), abstracts reviewed | Exp7746 measures independent retained groups after admission | Architecture reruns are deferred while local qualification fails; spline locality is no retention proof. |
| [FPGA–ASIC co-design](https://arxiv.org/abs/2602.15985), abstract reviewed; [Extropic Z1T](https://extropic.ai/writing/z1t), complete cost section read | Exp7750 measures host feature, calibration, decision, readout and durable acknowledgement; Exp7751 bounds hardware next steps | The Z1T projection excludes dense vocabulary readout and data movement; its own readout-inclusive estimate is much larger. Neither vendor projection nor existing board receipts are a Carnot acceleration measurement. |
| [HalluMix](https://arxiv.org/abs/2505.00506), abstract reviewed; [LagONN](https://arxiv.org/abs/2505.07179), abstract reviewed | Deferred until corpus custody or constraint extraction changes | No fresh cohort or new solver is introduced by this contract task. |

The bounded sequential delta check reopened the HalluSpan primary HTML,
delayed-capacity abstract, FPGA–ASIC abstract and vendor Z1T cost section.
The Semantic Scholar HalluSpan citation endpoint returned an access error.
Citation coverage is incomplete; no absence of later work is inferred.
