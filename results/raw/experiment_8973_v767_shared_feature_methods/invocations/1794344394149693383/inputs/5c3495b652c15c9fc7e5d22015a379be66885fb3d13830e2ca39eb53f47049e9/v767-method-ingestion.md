# V767 method ingestion — 2026-11-10

Exp8972 reads the V767 planning scan in research-references.md. Four primary
methods inform a development diagnostic. None supplies a local performance claim.
Exact retrieved HTML bytes and hashes live under
`results/raw/experiment_8972_v767_encoder_head_diagnosis/methods/`.

| Method | Equation and implementation requirement | Failure mode and local decision |
|---|---|---|
| [Message Passing Complexity](https://arxiv.org/html/2509.01254v2), Definition4.5 | MPC = -log P(lossyWL after L rounds retains enough information for f). Analysis needs a task, graph family and lossy-message model. | Limited message transport can lose information despite expressivity. Compare fresh heads on fixed features. This experiment does not estimate MPC. |
| [EnerGNN](https://arxiv.org/html/2610.33930), Sections3.2–3.4 | L = KL(p* || softmax(-E/T)) + lambda_cal L_cal + lambda_inf L_inf. Exact optimization needs an affine or linearizable readout, ReLU energy and valid MIP bounds. | Explicit constraints certify feasibility relative to those constraints. Solver cost and semantic truth remain separate. Reuse encoder/head separation only; add no solver. |
| [Energy-Based Reasoners](https://arxiv.org/html/2611.08977), Algorithms1–2 | z_next = Pi_M(RMS(z) + sigma xi - alpha grad_z E(z;x)). Training uses supervised descent segments, second derivatives and exploration across states. | A low learned energy need not imply truth. Trajectory supervision differs from Carnot's old generator protocols. Ingest the difference; keep the retired generator scope closed. |
| [LEF-JEPA](https://arxiv.org/html/2611.08989), Sections3.3–3.4 | q_next = q - alpha clip_tau(grad_q E(C,q)); L_enc = gamma softplus((E_bar(C,S) - stopgrad(E_bar(C,T)))/gamma). EMA targets and separate gradient paths train the field and encoder. | Visual representation gains and Ising timings do not transfer to graph answer verification. Use the frozen-evaluation and gradient-routing distinction only. No JEPA scope reopens. |

## Local protocol and limits

The existing encoder produces27 features: eight source coordinates, eight answer
coordinates, eight mean-node coordinates, two standardized confidence values and
one missing-answer flag. Each existing logistic and Gibbs encoder stays fixed.
Fresh linear and eight-unit Gibbs heads use the original seed initialization,
256 fit labels,200 full batches and rate0.01. The64 guard labels never train or
select anything. Temperatures stay one for the new diagnostic.

The logistic encoder candidate is chosen by this protocol before exposed scores.
All seeds71–73 remain in the candidate. No exposed-test gain can change this choice.
All V765/V766 labels are development evidence. The old128-graph metrics and five
failed bounds are reproduced with their original calibration and interpretation.
New predictions cover fit, guard and all1024 V766 graphs. Seeds and arms do not
increase the independent graph count or create new confirmation.

Encoder derivatives at frozen states describe sensitivity if unfreezing were
allowed. Actual encoder updates remain zero. Original joint-training derivatives
at steps0,1 and200 distinguish zero output initialization from persistent loss of
gradient. Later nonzero gradients refute a claim of permanently disconnected
gradients. They do not identify the causal reason for poor predictions.

Leakage controls reject truth-bearing input fields and preserve split roles.
Optimization comparisons hold feature bytes fixed within each pair. Differences
between encoders may reflect information or the historical training path; exposed
correlations do not prove a causal root cause. BFS truth keeps independent
oracle-free generalization at zero, regardless of diagnosis readiness.
