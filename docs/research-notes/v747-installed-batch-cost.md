# V747 installed batch cost

Exp8717 measures inference costs for the exact wheel and native extension
qualified by Exp8716. It uses the original 109-parameter numerical head and
64 exposed text requests. There are no language-model loads or calls. Native
execution is CPU execution; this study measures no physical FPGA acceleration.

The original V744 protocol supplies sizes 1, 4 and 16, ten paired process
repeats, sixteen panels per process, and seed 7448675. Paired condition blocks
are shuffled with that seed; each pair retains the registered alternating
Python/native order. Sixty fresh processes perform 61,440 item evaluations.
Repeated items measure runtime variation and add no independent semantic cases.

Complete throughput counts launch, imports, configuration, feature work,
packing, transport, scoring, parsing and clean shutdown. Offline wheel
installation has separate command receipts. Feedback updates and feedback
durability are outside this inference-only boundary. Requests have one frame
in flight; the result makes no interactive arrival or queueing claim.

Each response retains feature hashes, probabilities and typed decisions.
The parent retains raw frames, process exits and clocks. Scoring spans come
from the unchanged installed worker. Residual round-trip time includes wire
transport, serialization, validation and the extra feature-digest pass; these
costs cannot be separated with the existing wheel and are labeled together.
The disjoint components must sum exactly to the launch-through-exit clock.

The batching contrast is native B16/B1 complete throughput. Its registered
criterion is lower95 greater than 1.05, exact typed decisions, and pooled steady
item-completion p95 at most 50 ms. Each item in a batch completes when its
response is parsed. NFR-01 separately requires native/Python throughput lower95
at least 10 at the same batch size; B16 is the registered endpoint. All three
same-batch contrasts are retained. Intervals use 10,000 paired process bootstrap
draws. A batching gain is not evidence of a Rust arithmetic gain.

The primary result is `results/experiment_8717_v747_installed_batch_cost.json`.
Its raw invocation directory retains frozen sources and inputs, all sessions,
validation child SQLite and JUnit, command logs and sealed primitives. Cold
replay requires an independently supplied work hash and rebuilds the costs from
raw frames. Unchanged publication, adversarial and strict-row validators run
before atomic publication. Tests use private outputs; consumer checks do not
supply producer measurement or coverage credit.

OpenSpec records the owned reporting and verification requirements. The
conductor owns ops and BMAD reconciliation for this task.

Verification 2026-11-02, Exp8717: 60/60 process cells and 61440/61440 evaluations complete, with zero failed or censored cells. Cost readiness1; batching benefit1; same-batch10x0. Native B16/B1 complete throughput 1.1382x (95% interval 1.1218–1.1601), steady item p95 18.420ms. Native/Python B16 throughput 1.0054x (95% interval 0.9981–1.0139); NFR-01 is null. Eleven owned tests and two private E2E-018/021 reader tests pass. All394 newly owned statements are covered without exclusions; scoped Ruff, strict mypy, spec tracing, cold replay, missing/rehashed rejection, adversarial and strict row checks pass. This positive result concerns batching on one fixed CPU workload; it establishes no learning, independent generalization or FPGA acceleration.

Primary SHA-256: `sha256:938d2cd6a28f8b0ae1ee1a551199a8ab7c3b29de91dca12f2d83aa8d1379fef6`.
