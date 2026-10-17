# V740 Trivia+ private release custody

Exp8617, 2026-10-17, authenticates inputs for an internal, exposed coverage study.
It does not evaluate a detector, fit a head, or establish independent learning.
The author repository is
[amazon-science/hallucination-benchmark-trivialplus](https://github.com/amazon-science/hallucination-benchmark-trivialplus/tree/9416ae0c45ec4c4815c4b28820baa4016741deb7),
associated with [paper2605.11330](https://arxiv.org/abs/2605.11330).

The immutable release revision is `9416ae0c45ec4c4815c4b28820baa4016741deb7`.
README, DATA_DETAILS, LICENSE, the advertised checker and the parquet each match
the Git blob obtained from that revision's tree. Acquisition used seven bounded
requests, 15,004,699 bytes, and about 1.8 seconds of request time. The runner
authenticates these cached bytes again. Its current language-model calls, model
loads and numerical head evaluations are zero. Generator names describe imported
answers. The original nine-input spline head and checkpoint remain provenance.

The observed license is CC BY-NC-ND4.0. This task uses the release only for
internal noncommercial evaluation. It redistributes neither source nor transformed
text and infers no product-training permission. Original bytes and the predictor
projection stay in `/var/tmp/carnot-private-8617`, a private local cache outside
the results tree. Public artifacts contain hashes, counts and annotation summaries.

Raw human votes reproduce all 3,558 released sentence labels and all 3,224 binary
response labels. Ties prefer contradicted, then not-mentioned, then supported,
then supplementary. A response is unfaithful when any sentence is contradicted
or not-mentioned. The release contains 103 `n/a` annotation entries; missing and
surplus annotation positions remain visible. Automatic and noisy training labels
have no target authority. The local implementation is tested independently of
the acquired author checker.

The official split counts are 2,263 train, 316 valid and 645 test responses.
Normalization yields 2,270 question clusters and 1,786 complete-source clusters.
Fourteen question clusters span official splits. The registered Exp8615 rule
uses `seed7398607` to rank normalized question hashes without targets. It seals
128 of the 456 eligible test question clusters and retains all 190 sibling
responses across the release. A separate private predictor shard permits only
opaque ID, complete source bytes and answer bytes.

Historical manifests are inspected directly. Exp8403 and Exp8430 each contain
one overlapping question identity; these need not be distinct overlaps. Other
manifests lack comparable identities, so their historical overlap stays unknown.
Question disjointness and both generalization scores therefore remain zero.
Coverage-input readiness can still qualify this explicitly exposed study.

Validation collects original nodes before disjoint bounded test children. It runs
the new tests, private E2E-018/021 authority readers, primary publication tests,
Ruff, strict mypy, spec coverage, and Coverage.py on only the three owned files.
Cold children reject missing primitives and repaired-hash tampering. Final bytes
pass the unchanged adversarial verifier and strict row lint before atomic primary
publication. Exact commands, exits, durations, node union and log hashes are in
the artifact's validation receipts and bound terminal sidecar. Those receipts are
the authority for the final validation outcome.

No deadline behavior, learner, shared validator, generator weight, live roadmap
YAML, conductor or external service is modified. The Exp8603 disqualification,
V733 source null, V735 cold-cost null, V737 ARC null and V738 throughput limitation
remain historical evidence. Later V740 studies remain future dependencies. Ops
and traceability reconciliation is reserved for the conductor as instructed.
