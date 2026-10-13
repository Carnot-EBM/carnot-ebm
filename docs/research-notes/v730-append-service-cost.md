# V730 complete append service cost — 2026-10-13

Exp8494 compares the qualified experimental stable append store with the
unchanged snapshot service. Exp8492 seals the workload. Exp8493 qualifies
process-exit recovery. Each producer is authenticated by its original hash,
bound validation sidecar and original-authority replay before measurement.

The primary cell is natural batch1/reuse8. The secondary cell is batch8/reuse8.
Each cell has64 decisions, one warmup pair and five measured pairs. Arm order
alternates AB/BA. Fresh processes restore identical state on the same disk-backed
filesystem. Every complete clock includes cold numerical reconstruction,
startup, feature gathering, predictions, delayed feedback, checkpoints,
responses and final drain. Restart replay is separate.

All eight scheduled feedback events remain accounted for. The sealed natural
tape has seven applied updates and one nonapplicable feedback event. Both arms
must preserve bit-identical predictions, logical state and acknowledgment IDs.
The artifact retains each arm's clocks, latency, write bytes, sync counts,
successful namespace calls, peak resident memory and peak store file bytes.
The same observers run in both arms. Their elapsed cost is not subtracted.

Primary speedup is sum(snapshot wall)/sum(append wall). The confidence interval
resamples the five paired repeat indices10000 times with seed7298480. Benefit
requires lower95>1, primary p95 ratio<=1.05 and every durability/validation gate.
The secondary cell is descriptive. Incomplete pairs suppress both speed
intervals. Five repeats support conditional same-host engineering evidence.

All24 arms completed without a failed or censored repeat. Measurement took
211.77 seconds, below the1800-second cap. Warmups are excluded below.

| Cell | Complete speedup | Paired95% interval | Append/snapshot p95 ratio |
| --- | ---: | --- | ---: |
| Primary batch1/reuse8 | 2.338634 | [2.162220,2.529340] | 0.517040 |
| Secondary batch8/reuse8, descriptive | 1.466303 | [1.176385,1.908085] | 0.810219 |

The primary timing operands satisfy lower95>1 and p95<=1.05. Readiness also
requires the terminal artifact's independent replay, coverage and validator gates.
The host tenfold target remains open because the primary lower bound is below10.

Each primary snapshot arm performs146 file syncs,146 directory syncs and293
successful namespace calls. Each primary append arm performs157 file syncs,15
directory syncs and24 successful namespace calls. The append mechanism reduces
directory and namespace work while retaining every feedback event and reply.
The secondary snapshot counts are34/34/69; append counts are41/9/14.
All repeats have eight scheduled feedback events, seven applied updates, one
nonapplicable event, and zero lost, missing or pending events.

The [primary artifact](../../results/experiment_8494_v730_append_service_cost.json)
contains terminal readiness and every measured repeat. Primitive evidence and
validation receipts remain under
`results/raw/experiment_8494_v730_append_service_cost/`.

Exp8466 remains the historical different-mechanism comparator: qualified
snapshot/journal speedup1.0447833367,CI95[0.9942346768,1.1095895448]. Exp8494
does not rerun or replace that null. V726 exposed-history results remain intact.
V717 H1=-0.00390625/H2=0 remain closed. Generator weights remain frozen.
No model is loaded; current LLM calls and both generalization scores are zero.
Physical power-loss certification and matched Rust/Python NFR-01 remain open.

The runner collects scoped tests before execution, partitions historical
E2E-018/021 suites without overlap, and retains command/exit/duration/log hashes.
It combines genuine unit/child Coverage.py shards for only its newly owned
statements. Unchanged publication, adversarial and strict-row consumers check
the terminal bytes. Conductor reconciliation owns ops and traceability files.
