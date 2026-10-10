# V722 external evidence readiness — 2026-10-10

The released evidence does not qualify an independent semantic corpus.
The inspection preserves96 slots:32 per lane. It opens32 released HotpotQA
examples. Grounded QA and executable-code output releases remain absent.
Both generalization scores stay zero. The run loads no model and makes no
generation or paid annotation calls. No other V722 task depends on this result.

## Pinned author releases

[OpenHalDet](https://github.com/Nellie179/Hallucination-Detection), for
[2606.06959](https://arxiv.org/abs/2606.06959), is pinned at
`82a8c18574fe5706d69237abc716d2361f4bd3f8`. Its tree contains46 files.
It provides adapters and generation/annotation code. It has no release assets,
generated outputs, hidden states, or scored metadata. Its README claims MIT;
the linked LICENSE file is absent. Dataset licenses remain separate.
The default pipeline uses a model judge. Those labels are weak labels.
The run does not invoke that pipeline.

[Verification Without Sufficiency](https://github.com/iamhero2709/verification-without-sufficiency),
for [2608.00585](https://arxiv.org/abs/2608.00585), is pinned to the author
paper tag `v1.0-arxiv`, commit `09c983803e0eeea41d7fa2b9b3d8f80d7b0f8ea8`.
The current HEAD is recorded separately. The paper tag contains splits,
per-question traces and analysis. Code is MIT. Dataset terms remain separate.
[HotpotQA](https://github.com/hotpotqa/hotpot) specifies CC BY-SA4.0 for data.
The raw manifest records actual tree paths, sizes, blob IDs and byte hashes.
Retrieval used4,578,657 bytes in357 seconds, below200 MiB and15 minutes.

## Lane decisions

| Lane | Released observations | Independent target authority | Decision |
| --- | --- | --- | --- |
| Grounded QA | OpenHalDet has adapter code; its generated/scored files are absent. | Original RAGTruth human labels require separate acquisition and overlap checks. Model judges remain weak. | Defer corpus selection. Adopt the read-only inventory. |
| Multi-hop QA |32 HotpotQA split records and paired Qwen2.5 B1 outputs. | Gold QA answers and supporting paragraphs exist. EM/F1 is a response proxy, not human semantic correctness. | Adopt structural replay. Defer detector evaluation. |
| Executable code | HumanEval/MBPP adapters exist; generated outputs and execution receipts are absent. | Independent tests can supply bounded execution authority after safe acquisition. | Defer acquisition and benchmark. |

Carnot already contains RAGTruth and HumanEval/MBPP evaluation manifests.
Their metadata files are hash-bound. Exact question, document and code-task
overlap remains unknown. Dataset presence cannot establish cluster disjointness.
The32 multi-hop examples identify clusters by paragraph-title sets. This does
not certify independence between overlapping Wikipedia sources.

Selection uses SHA256 of `7228375|hotpotqa|ordinal`, in ascending order.
The author metadata supplies500 ordinal slots. Outputs and labels do not
choose the32 inspected records. The adapter preserves every unavailable slot.
The release supplies source paragraphs and gold supports. Generation traces
do not supply the exact retained paragraph IDs. Single-evidence availability
therefore cannot be paired with complete evidence. No semantic comparison is
reported. Gold supporting evidence and decomposition remain oracle-only.
No per-claim human labels are supplied. Availability does not establish truth.

## Future sampling and acquisition

Before opening a new locked evaluation, authenticate at least80 disjoint,
independently labeled source clusters, with8 clusters per class. Count support
from metadata. Separate train, tune and evaluation source roles. Hold out a
generator family. Exclude exposed V717 sources and all authenticated overlap.
Do not replace missing releases with FoVer. Preserve missing denominators.

The planning budget starts with120 clusters and two outputs per cluster.
Use `unsloth/Qwen3.8-27B-GGUF`, Q4_K_M, only after authenticated runtime
preflight. At512 tokens per output and an assumed20 tokens/second, generation
uses1.707 GPU hours. Preflight and evidence/I/O allowances bring the budget to
3.207 GPU hours. An assumed$1/GPU-hour gives$3.207 of compute. These are
planning assumptions, not measured local performance or current quotes.

Two independent human reviews totaling10 minutes per output require40 hours.
At an assumed$25/hour, annotation costs$1,000. Executable-code labeling has a
separate two-CPU-hour cap and needs isolated test execution. Total estimated
compute plus human annotation is$1,003.207, before adjudication or storage.
This note grants no permission to spend money or load models. Future acquisition
and label-ledger paths are dependencies. Current LLM calls remain zero.

## Validation boundary

Private positive, missing-field, contaminated-source and wrong-authority controls
test parsing. Private E2E-018 tests matching authority, mutation, deletion and
reordering. Cold children exercise valid, missing-input, deliberate-error and
self-consistently rehashed tamper cases. The unchanged publication, adversarial
and strict row consumers check terminal bytes. Owned failures disqualify.
Unchanged external evidence absence blocks. Full repository health stays in a
separate receipt. Ops and traceability reconciliation belong to the conductor.
