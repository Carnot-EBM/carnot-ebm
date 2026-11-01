# V746 fixed-point lifecycle stress

Exp8706 uses the original five Exp8431 spline heads and Exp8692's frozen256
vectors and four deterministic label/order sequences. It preserves the Brier
update, temperature and scaler, signed32/24 format, ties-to-even rounding,
explicit saturation and checked signed64 intermediates. There are no LLM calls.
The twenty ordinary arms each run4096 updates against float64 (81920 pairs).
A separate matched integer replay exits after committed2048 and resumes from
serialized state in a fresh process. State and acknowledgement must match.
Replay work is dependent work and adds no independent questions.

Five independently marked boundary states set their decay integers to65534
through65538, with the original integer coefficients. They run64 updates each.
Their float64 counterparts start from equivalent effective coefficients. These
320 controls are constructed, not naturally learned states. Before/at/after
trigger rows retain the original rounded-next-scale<65536 condition. Dense109
renormalization writes, local<=37 data writes, global decay,109 host snapshot
copies, CPU nonlinear work, serialization, durable writes and restart work stay
in the cost boundary. Integer update timing includes any dense renormalization;
it does not isolate or subtract its cost.

Every ordinary step requires finite values, zero saturation/overflow and
probability error<=0.001. Typed decisions agree outside that margin; inside it,
the original float64 fallback is executed and counted. Boundary controls check
effective-coefficient continuity to half a stored coefficient quantum. A failed
numerical criterion produces a null with the first failing state, without
retuning. Complete qualified rows can be ready even when numerical criteria fail.
The artifact contains the measured outcome and immutable raw invocation paths.

The conditional accumulator envelope assumes37 terms, signed32 coefficients,
basis weights bounded by2^24+16, scale within[2,2^24], temperature within[0.5,2]
and finite scaled inputs within[0,1]. It bounds the residual sum by
37*2^31*(2^24+16), with checked intermediate rejection retained. The looser
whole-format37*(2^31)^2 bound exceeds signed64. Neither observation nor this
supported-domain envelope is a global overflow proof.

[Hoang et al., arXiv:2602.02056v4](https://arxiv.org/html/2602.02056v4)
motivates explicit formats, rounding, saturation and locality. This study is
CPU emulation. It establishes no useful learning, independent generalization,
physical FPGA execution or acceleration. Historical quarantines, Exp8681 and
Exp8691 disqualifications and V733/V735/V737/V743 nulls remain unchanged.
Ops and traceability reconciliation is conductor-owned.
