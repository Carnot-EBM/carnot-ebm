# V744 ARC observation ownership

Date: 20261101. Producer: Exp8677. Requirements: REQ-REPORT-8677 and
REQ-VERIFY-8677. The terminal result is
`results/experiment_8677_v744_arc_observation_ownership.json`. Read its
`observer_delivery_ready_score` and validation sidecar for qualification.

The old path delivers one returned observation through E3AgentPolicy.is_done,
CarnotAgent.choose_action and E3AgentPolicy.next_move. The collector delivers
the final return. For512 actions, the first511 returns have two extra
deliveries:1022 duplicate_observation issues. The experiment loads the sealed
original wrapper and replays each of the sixteen original public SDK frame
traces to reproduce this count. It does not inspect game implementation.

The repair adds one ingress ledger to the installed observer. The first live
callback owns delivery for that action index. Every later callback checks a
digest of the complete public frame, including pixels. Identical callbacks
are retained as duplicate_callback rows. Changed pixels or metadata for the
same action produce conflicting_duplicate and censor pending horizons.
Identical pixels at different action indices remain separate outcomes.

ArcOutcomeObserver and its duplicate/conflicting_duplicate assertions remain
unchanged. Receipt errors remain visible and cannot replace policy actions.
The repair changes no action selection, quota, horizon, supervisor priority,
generator configuration, weights or game knowledge.

The methods protocol registers four games, seeds11/23 and recorder off/on:
sixteen fresh official offline SDK episodes through make_carnot_agent and
E3AgentPolicy. The older diagnosis uses the original seeds11/22. Registry
precheck retains already reproduced levels before execution. Policy inputs
withhold routes, solutions and per-game adapters. Language-model calls are
disabled. The small numerical action-effect checkpoint is authenticated
separately; no_model_load does not imply no numerical inference.

Exact parity compares actions, observations, final states, termination checks,
reset causes, random states and supervisor receipts. Constructed controls
include a completed120-action horizon, reset censoring, terminal censoring,
repeated is_done and pixel-only conflict rejection. A separate test calls the
actual policy is_done repeatedly. These controls establish counting behavior,
with no natural supervisor benefit or solve credit.

The runner seals measurement and validation separately. Cold replay rejects
missing operands, source drift, changed logs and self-consistently rehashed
summary or receipt substitutions. Raw primitives, source snapshots, SQLite,
JUnit, collected node IDs and bounded child receipts remain under the raw
invocation directory. Primary publication checks cold replay, adversarial
verification, strict row consistency and both unchanged consumers.

Exp8664 remains flagged and disqualified. Exp8667's declared ARC gate is1,
while its global quarantine remains preserved. All earlier quarantines and
failed receipts retain their original bytes. Routing readiness cannot promote
policy benefit, generalization, stored solves or headline solve credit.

During development, two checks shared a scratch directory and one lost a log.
The isolated run passed all16 tests. Final qualification uses separate child
directories and retains its own validation receipts. Ops, status, changelog
and traceability reconciliation are conductor-owned.

The first real SDK measurement attempt failed serialization: the SDK returned
a list containing NumPy arrays. The earlier FrameData fixtures used nested
Python lists. The stopped attempt retains eleven failed child receipts, its
source snapshots and an explicit abort record, with zero qualification credit.
A new regression checks full64x64 SDK pixels, JSON serialization and a changed
interior pixel that a truncated array string could conceal. The corrected
ingress converts the complete pixel array to plain lists before hashing.
The final invocation uses a new source seal and retains the failed attempt.
