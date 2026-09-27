# V676 bounded Qwen confidence protocol

The frozen task protocol is
`results/raw/experiment_7770_v676_qwen_runner_qualification/protocol.json`.
It names the exact 24 exposed Exp7745 families. The source and answer bytes
remain identical between arms. Both arms use seed 67670, temperature zero,
disabled thinking, one strict JSON schema, and at most 256 output tokens.

The generic arm asks for answer correctness probability. The reducer maps its
reply to unsupported risk with `1 - p`. The event arm asks for the probability
that at least one answer claim lacks support in the supplied complete source.
These targets differ. A comparison can report their disagreement, but cannot
call them equivalent calibration targets.

The shared schema requires a finite probability in `[0,1]`. It permits an
optional list of evidence sentence ids. Invalid, missing, or truncated output
keeps its family in the denominator with risk `0.5` and forced escalation.
No label selects a family. The 24 families are exposed development evidence.

Exp7759 retained raw replies that show the old parser failure. Its canonical
arm parsed 1/24; paired parsed 2/24; source withheld parsed 9/24. Only one
family parsed in both complete source arms. For example, the canonical and
paired replies for family `sha256:a66ce67c7ee5ad2d15c796bd92b3fcb5203b5db0361dcc880a1b198694efd1fe`
begin with a fenced `json` block. The same family's source withheld reply is
also fenced and says abstain. The raw files are in
`results/raw/experiment_7759_v675_qwen_evidence_views/runs/1790505190-1914157/`.
The strict parser rejects fencing. The result is a parser failure, not a
calibration or invariance finding. Exp7759 remains disqualified.

The CPU fixture uses the same `/v1/chat/completions` streaming request shape as
the local llama.cpp server. The installed server source accepts
`response_format.type=json_schema` and reads
`response_format.json_schema.schema`. The fixture rejects a request that omits
that schema. This checks transport and reducer behavior without a model load.
It does not test Qwen generation or natural probability quality. Future live
capture must authenticate the real server identity and retain raw responses.
