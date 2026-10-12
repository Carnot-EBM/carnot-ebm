# V728 objective and checkpoint qualification

Exp8461 qualifies the Brier update and durable constructed feedback before a
separate task measures learning. It loads no language model. Generator weights
stay frozen. The inherited V726 head has nine inputs and109 coefficients.
The adapter keeps its scaler, temperature, local basis support, rate0.01 and
L2=0.001. The bias receives the same L2 penalty as every other coefficient.

The data derivative is `2*(p-y)*p*(1-p)/T` times the basis value. The inherited
log-loss derivative is `(p-y)/T`. Individual cubic splines provide an independent
dense reference. Centered differences perturb effective coefficients by1e-4.
This step avoids cancellation in the regularized objective at saturated logits.
The absolute and relative limits remain1e-7 and1e-5. The relative limit applies
when the reference magnitude exceeds1e-6. Both labels, three temperatures,
five logits and three basis supports produce90 constructed cells.

A small descent step must reduce the full Brier-plus-L2 objective. Small data
gradients at saturation are reported explicitly. They do not establish learning.
Temperature omission and gradient reversal are checked against the independent
reference. Future labels must fail the sealed admission roster check.

The adapter inherits the store's immutable snapshots, writer lock, file flush,
atomic pointer replacement and saved acknowledgments. It changes numerical
release and checks the update-rule version, rule hash and implementation hash.
Transaction one seals all sibling predictions and admissions. Transaction two
saves released updates, feedback identities, both random states and the cursor.
Real private children exit73 before and after both commits. Resumed predictions
and complete logical states must equal uninterrupted execution bit-for-bit.
These checks do not recertify physical power-loss behavior.

Current task authority comes directly from the full activated V728 contract.
The historical head and store are authenticated under sealed original V726
producer authority. Optional CUDA, board evidence and absent V727 science are
not CPU prerequisites. All adapter updates use constructed labels. Historical
producer replay remains retrospective evidence.

`objective_kernel_ready_score` requires numerical, recovery, authority and
owned validation checks. `objective_stream_ready_score` stays0. Generalization
and learning-benefit scores stay0. The raw evidence includes the adapter bytes
and a fixture receipt for the separate measurement task. Validation uses scoped
serial tests, genuine child coverage, private E2E-018/020/021 and unchanged
publication, adversarial and row consumers. Cold replay reconstructs primitives
and rejects rehashed tampering.

Historical findings remain unchanged: V727 had six blocked primaries and eight
skipped slots, with methods readiness0. V726's admission-learning audit was null.
Kernel arithmetic was0.093 percent of service and persistence was96.44 percent.
V717 H1=-0.00390625 and H2=0. Public-label evidence remains retrospective.

The terminal result is `results/experiment_8461_v728_objective_checkpoint_qualification.json`.
The conductor owns status, changelog and traceability reconciliation.

The publication run completed all90 derivative cells and36 recovery cases.
Its38 owned tests passed. The unchanged historical regression command reached
its600-second deadline and was killed. This owned validation failure disqualifies
readiness. The kernel readiness score stays0 despite the numerical and process
checks passing. The terminal artifact retains that failure and subsequent
validation receipts. This result supports no learning-benefit claim.
