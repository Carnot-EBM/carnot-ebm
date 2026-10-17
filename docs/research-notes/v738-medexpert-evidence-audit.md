# MedExpert original and assisted evidence audit

Exp8597 audits annotation authority before any independent verification claim.
Its primary is `results/experiment_8597_v738_medexpert_evidence_audit.json`.
The direct CPU runner accepts `--date 20261017` and other valid calendar dates.
It loads no language model, energy head or checkpoint. Current LLM calls are zero.

## Observed sources and limits

The [analysis paper](https://arxiv.org/html/2609.30467v1) links an anonymous
repository. Its landing route redirects to an API file route. That route returned
HTTP 401 without credentials. This audit does not bypass that response.

The [original publication](https://proceedings.mlr.press/v297/yarmohammadi26a.html)
links [JHU-CLSP/MedExpert](https://github.com/JHU-CLSP/MedExpert).
Its acquired Git tree is `98d19d9ec7f0b03273b97a9d6bf5e71548c98ede`.
This is a tree hash, not an authenticated commit hash.
The exact README blob matches the recorded original tree.
That README links the original
[Hugging Face release](https://huggingface.co/datasets/sonal-ssj/MedExpert).
Its acquired metadata identifies commit
`1ada617c799237fa2c5d3e71e4d2c9b0a5857fb3`, a public ungated dataset,
CC BY 4.0 terms, and two JSONL filenames.
Metadata availability is separate from acquired sentence bytes.

The invocation used eight bounded HTTP transfers and two browser reads of the
same analysis paper. Each HTTP transfer had a 30-second timeout and a byte cap.
The ten-request limit ended retrieval. No original or assisted sentence file was
acquired. The original files are discoverable; their contents remain unaudited.
The assisted release has an observed external access block. All HTTP bodies,
headers, statuses, URLs, byte hashes and transfer durations are retained under
`results/raw/experiment_8597_v738_medexpert_evidence_audit/`.

## Annotation meaning

The original paper reports 540 question–response pairs from 108 questions and
five generators. Eight practicing clinicians annotate factual error spans and
response omissions. This documents original clinician authority. It does not
prove per-row authority, blinding, or independent correctness of unmarked text.
The imported generator names appear only in `historical_model_provenance`.

Appendix A.2 of the analysis paper selects 225 originally error-marked sentences.
Five adjudicators see sentence-level claims, six verifier votes and reasoning.
The paper reports 86 retained errors and 139 sentences reclassified as correct.
These are documentary counts, not counts verified against release rows.
Table 1 lists 224 original errors. The 224/225 discrepancy remains unresolved.
The assisted labels are not independent of the shown verifier outputs.

The target criterion marks a sentence erroneous when any associated claim is
false. Evidence availability and passage relevance have different meanings.
Mapping No Evidence or Ambiguous to a binary class does not establish truth.
Neither unmarked sentences nor unavailable passages become verified negatives.

The selected subset cannot estimate population precision, natural prevalence,
or general-domain accuracy. Inclusion is conditional on an original error mark.
Question duplication, natural sampling probabilities, span offsets, missing
review flags, error types, row generator identities and gold rights remain
unverified without row bytes.

## Eligibility and future work

A completed bounded audit can have readiness one and evaluation eligibility zero.
This audit records separate release availability, original annotation authority,
assisted independence, question disjointness and rights gates.
Missing target questions leave every overlap count unknown. The authenticated
registry includes LPHB, HalluTruthQA, the current learning exposure ledger,
HALLMARK and the local exclusion manifest. No empty inventory is called disjoint.
No eligible target inventory is sealed. No detector is fitted or scored.
Unknown generator-training contamination remains unresolved.

Future evaluation requires original and assisted row bytes, explicit rights,
blinded adjudication and a sealed question partition. Normalize question/source
strings with NFKC, case folding and whitespace collapse before hashing.
Audit all rows up to the 15000-row cap without publishing patient text.
A negative match addresses only locally recorded exposure. It does not establish
absence of training contamination. Population estimates require prospective
sampling outside this selected subset.

V737 completed twelve scheduling slots. Exp8580 qualified the current validator;
Exp8581 qualified causal accounting. Exp8582 runner-clock numbers and Exp8584
methodology-provenance numbers remain quarantined. Exp8588 remains a qualified
ARC null without refinement support. The completed archive ends at V736.
Source and cold-cost nulls, old hashes and label rejections remain unchanged.
Same-roadmap outputs are future dependencies. No medical advice or detector
benefit follows from this audit. The conductor owns ops and traceability updates.

## Validation evidence

All44 collected nodes passed: seven owned tests, eighteen original E2E-018
nodes, seven original E2E-021 nodes and twelve publication-consumer nodes.
Original test files and authority dates remain unchanged. Disjoint child JUnit
reports reproduce the exact original node union. No child was killed.
Scoped Ruff check and format, strict mypy and spec coverage passed.
Combined unit and genuine CLI coverage is160/160 newly owned statements.
The runner cold-replays original operands and runs the unchanged adversarial
scanner and strict row linter before atomic primary publication. It retains
scanner findings, exits, durations, stream hashes and actual reader acceptance.
The full repository suite and historical validation controllers were not run.
