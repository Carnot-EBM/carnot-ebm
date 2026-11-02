# Resident sparse update prototype

Exp8731 adds an opt-in `spline-resident-8731` feature. It uses the existing
signed32/24 checked integer kernel. One PyO3 object owns109 coefficients, decay
scale, temperature and arithmetic counters. The default learner and service do
not change. Host scaling, log/logistic work, finite checks, label admission,
durable checkpoint writes and acknowledgements retain their original roles.

Typed preparation returns local indices, weights, a residual and a one-use
preparation token. Typed application consumes the current token and returns
only decay scale and sparse/dense write counts. Duplicate or superseded tokens
cannot acknowledge an update. Host checkpoint import recreates state and
counters; pending probability work must be prepared again after process restart.
The host continues to own external feedback identifiers and durable admission.

The qualification driver is an isolated binding of the existing Exp8719 driver.
It compares all81920 original updates and320 boundary controls with the Python
emulator and hash-bound original native witnesses. Real workers checkpoint,
exit73 after committed update2048 and resume in new processes. Serialization
keeps little-endian coefficient, decay, temperature and host scaler order.

Qualification reads counters and coefficients to assert exact parity. These
diagnostic copies are recorded separately from typed preparation/application.
Byte rows describe logical field payloads and explicit vector copies; they do
not measure CPython object allocation or timing. The original update kernel
still clones109 coefficients internally before committing its result.

The supported accumulator envelope remains1333066761011986432. The whole-format
bound170632382681813352448 remains unsupported. Zero denominator, NaN, signed64
overflow, saturation, ties-to-even, negative floor division and dense decay
renormalization retain owned regression paths.

The primary result and its raw invocation directory retain actual test exits,
JUnit, coverage SQLite, source/input hashes, clocks, loaded binary identity and
terminal validator receipts. Sources, inputs, test identities and methods are
sealed before execution. Private fixtures cannot grant production readiness.
Missing external operands block; failed owned checks disqualify. Cold replay
and the unchanged publication, adversarial and strict-row validators precede
atomic publication.

This experiment proves numerical and lifecycle eligibility only. Exp8732 owns
performance measurement. Exp8717's1.138x batching and1.005x same-batch comparison
retain their qualifiers. No model load, generation, new ARC outcome, CUDA repair,
causal learning benefit or physical acceleration is claimed. Historical primary
results remain unchanged. The conductor owns ops and traceability reconciliation.
