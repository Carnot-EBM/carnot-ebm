# V737 HalluTruthQA release audit — 2026-10-16

The complete author release was inspected locally without model execution.
Audit completion establishes an inventory, not independent generalization.
The report is `results/experiment_8585_v737_hallutruth_release_audit.json`.
Raw release text and per-instance diagnostics remain in private `/var/tmp`
storage. Public primitives contain hashes and aggregate diagnostics only.

The pinned [author release](https://huggingface.co/datasets/Bekhouche/HalluTruthQA-4K/tree/e6e2b14733effc8da7b87fa79230a20cc4b7336b)
contains 4,000 instances: train 1,600, development 800 and test 1,600.
Downloads of the six repository files total 7,684,025 bytes. Paper pages and
public API metadata are separately hashed. Original split assignment is kept.
The complete downloaded JSONL files are retained without edits.

| Split | Hallucination | No hallucination | Exact duplicate question excess |
|---|---:|---:|---:|
| Train | 726 | 874 | 0 |
| Development | 380 | 420 | 0 |
| Test | 683 | 917 | 2 |

All 2,419 released spans match character offsets in unchanged UTF-8-decoded
responses. Byte-offset agreement is diagnosed separately. There are 1,789
hallucinated instances. Exact question overlap is one train/development hash,
two train/test hashes and five development/test hashes. These are overlap
counts, not independent observations. No records were removed or reassigned.
The paper's train/development count of 1,037 hallucinated responses differs
from the release's 1,106; the audit retains the release labels.

The [corpus paper](https://arxiv.org/html/2608.03966v1) documents domain experts,
two independent research assistants and expert resolution of disputed cases.
This is human-label evidence independent of Carnot scores. It is not a release
of individual adjudication records. All 4,000 `needs_human_review` flags are
false, but `prepare.py` defaults missing flags to false. No original per-row
annotator, reviewer or adjudication logs accompany this release. Table 4 also
qualifies agreement figures as requiring confirmation against complete logs.
The verdict combines factual correctness and evidence/source attribution;
it is neither pure reference entailment nor exact answer-key matching.

All responses come from the historical QCRI/Fanar-1-9B-Instruct generator.
The [shared-task paper](https://arxiv.org/html/2609.38355v2) distinguishes unseen
questions from unseen generators. This single-generator controlled release
cannot establish unseen-generator performance or natural deployment error
prevalence. All official splits are now exposed to this audit. No detector
was trained and no test labels selected features.

The pinned card states [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
The audit establishes no permission for intended product deployment. It makes
no translation, redistribution or modification of raw text. The data-only
exclusion manifest removes labels, references, answer keys, spans, options,
review notes and construction metadata. Its tested prospective reader retains
only question, generated response and separately permitted evidence.

| Primary source | Adopted operation | Deferred claim |
|---|---|---|
| HalluTruthQA-4K, 2608.03966v1 | Verify author label lineage, immutable spans and full split inventory | Independent factual benchmark, row adjudication certification, multilingual transfer |
| HalluScoring, 2609.38355v2 | Separate question holdout, generator holdout and sampling regime | Treating Task 1 generalization results as HalluTruthQA Task 2 evidence |
| [Calibeating, 2603.22167](https://arxiv.org/abs/2603.22167) | Preserve proper-loss comparisons and prediction before feedback | Regret theorem or label-authority transfer |
| [Online KAN, 2602.02056v4](https://arxiv.org/abs/2602.02056v4) | Retain sparse update and complete service-cost boundaries | Published FPGA timing as a local measurement |
| [KAN sample efficiency, 2610.10627v1](https://arxiv.org/abs/2610.10627v1) | Keep fixed optimization prefixes and matched controls | Inferring independent generalization from training curves |

Previously scanned proper-loss and KAN methods are bound through the original
Exp8513/Exp8571 literature bytes. Their mathematics does not create human truth
labels. No model, generator weights, training corpus or active roadmap changed.
The HALLMARK rejection, Exp8529 exposed-source null, Exp8559 cold-cost null,
V736 authority failures and all existing quarantines remain in force.

A later evaluation requires fresh unexposed question clusters, independent
adjudication, a language-qualified feature pipeline and rights for intended
use. Generator holdout additionally requires withheld generator families;
natural prevalence additionally requires prospective deployment sampling.
No multilingual protocol is admitted on this Arabic-only evidence. All
same-roadmap outputs remain future dependencies.

The binary `natural_error_evidence_score` is Boolean false (closed/zero).
It denotes eligibility and makes no zero-error measurement. The initial
numeric-zero prototype's informational scanner finding is retained in raw
validation history. Required terminal checks use unchanged shared validators.
