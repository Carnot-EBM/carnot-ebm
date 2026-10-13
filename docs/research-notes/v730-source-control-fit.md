# V730 source control fit — 2026-10-13

Exp8487 fits the missing q-only cubic energy head after Exp8486 qualified the
cached source controls. V729 skipped this fit because its authority was incomplete.
That skip provides no result for the source-information hypothesis.

The runner authenticates the full activated V730 task contract. It requires the
qualified Exp8486 primary and its bound publication report. Independent V726
producer hashes bind the original trainer view and frozen heads. Method hashes,
fit scaler and knots must match before the comparison can proceed.

The five arms have 0,2,13,109,109 parameters: raw p2,scalar logistic,q-only cubic,
full cubic and source-permuted cubic. Four arms retain original head bytes.
Only the q-only cubic is trained now. Its twelve one-input cubic basis
coefficients and bias start at zero. The fit uses200 full-batch steps,rate0.01
and L2=0.001. The original logit(q) offset remains part of the energy.

All five seeds produce the same scalar coefficients. They represent one
deterministic fit,not five independent replications. Each seed retains its
actual optimization clock. The old producer omitted individual head timings;
those imported timings remain missing with an explicit reason.

The scalar fit objective falls from0.3519971699730196 to0.35141587822538645.
Tune log loss selects T=0.5: losses are0.25487044918640206,0.25696651060177716
and0.382264876663633 for T=0.5,1,2. Independent dense bases agree exactly.
Finite differences check both labels,all three temperatures and global L2.
Omitted-temperature and omitted-regularization controls fail those checks.

The runner writes frozen heads before predicting300 test answers across five
arms and five seed slots. It retains7500 label-free typed decision rows.
E(0)=0,E(1)=-eta/T and p=sigmoid(eta/T). Fixed actions accept p<=0.2,
reject p>=0.8 and otherwise escalate. Confidence is max(p,1-p).
The fit and predictor open no test-target file.

Fresh bounded cold replay reconstructs the scalar fit from independently
sealed operands. It compares exact float64 prediction bytes. Primitive
coefficient,decision,authority,log and signed-zero drift reject even when
their enclosing hashes are repaired. Genuine Coverage.py subprocess shards
measure only newly owned statements. The runner collects historical test IDs
before bounded nonoverlapping E2E-018/021 shards execute. Each command retains
its exit,duration and stream hashes. Unchanged publication,adversarial and
strict row consumers precede atomic publication.

`head_controls_ready_score` qualifies fitted controls. It does not measure a
scientific benefit. Human targets and cached judgments remain exposed.
Both generalization scores stay zero. There are no model loads,LLM calls,
generator weight changes or optional hardware gates.

V728 qualified journal equivalence remains unchanged. Its complete-service
speedup is1.0447833367 with CI95[0.9942346768,1.1095895448],a null benefit result.
V726 exposed-history findings remain unchanged. V717 H1=-0.00390625 and H2=0
remain closed. Exp8488's information audit is a future dependency,not an
observation of this fit.

The conductor owns later ops/status/changelog/traceability reconciliation.
