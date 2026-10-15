# V733 ARC capability qualification

Exp8536 adds an opt-in availability predicate to the live E3 supervisor. The
flag is `CARNOT_ARC_SUPERVISOR_CAPABILITY_MASK=1`. The policy reads the current
induction gate for each action. When induction is explicitly disabled,
`allow_reinduction` and `tool_loop_reinduction` cannot be selected. Their
baseline predicates and capability rejection remain visible. They do not enter
the tried set. Unknown capability keeps its normal eligibility and cannot show
a mask benefit. A later runtime change restores normal eligibility.

The default path keeps the original snapshot and receipt shape. The curated
arm order, window120, evidence floor200, attempt cap3 and stall threshold150
remain fixed. The diversity latch is preserved across levels. An unredirected
window distinguishes `no_currently_executable_arm` from `all_arms_attempted`.
The existing direct receipts keep selection, latch changes and effective
search changes separate. The report gives each arm its own baseline predicate,
capability predicate, reason, selected status and applied status.

The motivation is the original Exp8523 artifact from20261014, sealed at
`sha256:26d4935b96524fa0ce12af38f8bfae4fb823e848f85bb0afbc68ce97881e73fc`.
Its eight diversity applications changed effective search. Its eight
reinduction latch changes reached `disabled_by_env` without a model action.
All32 redirects were censored. These observations justify an availability
correction. They do not establish an arm ranking or strategy benefit. This
qualification authenticates the original authority, bound validation and
primitive hashes; it does not rerun the historical validation campaign.

Private fixtures use the actual `make_carnot_agent` factory and
`E3AgentPolicy.choose_action`. Cases cover disabled, enabled but uncalled,
unknown and changing capability, already-active diversity, level transitions
and shadow mode. Receipt-on/off comparisons include full returned action
payloads, state, provenance and random state within each policy. An original
policy method from the preserved source also checks default-off byte parity.
A mutation that ignores model absence fails the disabled-selection assertion.
Tripwires prevent any real model load or network call.

The frozen future panel contains ar25-0c556536, ft09-0d8bbf25,
lf52-271a04aa and tn36-ef4dde99 with seeds11/22. Each pair compares
`baseline_applied` with `capability_masked_applied`, using512-action and
120-second episode caps and the original stop rules. The registry is read
before outcomes and is never provided to the policy. Adapters, stored routes,
solver artifacts and game source introspection are withheld. These16 slots
belong to Exp8537. Exp8536 measures no live episode and grants no solve credit.
Readiness qualifies code and the frozen protocol only. Both generalization
scores and headline solve credit remain zero.

The runner collects original test node IDs before executing disjoint bounded
serial shards. Private E2E-011/013 exercise wrapper telemetry; private
E2E-018/021 exercise original authority and publication consumers. Owned unit
and subprocess coverage is separate from added-statement coverage in the two
existing live modules. Cold replay checks fresh wrapper primitives and rejects
absent, failed and self-consistently rehashed evidence. The unchanged primary
publisher and terminal validators bind exact checked bytes before the actual
downstream consumers read the readiness field.

The owned verification scope is24 tests and360 statements across five new
files, plus13 added statements in the two existing live modules. Coverage is
measured from unit execution and real CLI children. Original assertions are
kept unchanged. Development logs retain the initial expected failures and the
first historical-reducer memory-guard failure. The correction uses sealed
historical custody and direct-row authentication; it does not weaken that
memory guard or any scientific threshold. The terminal sidecar is the authority
for final checks and consumer selection.

Run from the repository root:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8536_v733_arc_capability_qualification.py --date 20261015
```

The private selection fixtures use a one-action stagnation window to exercise
the table directly. They retain the evidence floor, attempt cap and diversity
stall threshold. The unchanged native configuration and frozen future panel
use window120; fixture timing is not a live timing or strategy measurement.
