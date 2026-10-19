# V743 real SDK reset provenance — Exp8663

Exp8650 remains disqualified for provenance. Its immutable public streams retain
236 resets without authenticated causes, 688 request rows, 16 censored redirects
and eight exact action-parity pairs. Exp8649's scripted fixture qualification
remains valid within its original scope. No historical null or quarantine changes.

The retained public reset pixels reconstruct the real SDK representation:
`FrameDataRaw.frame` is a list containing NumPy arrays. The old recorder converted
only an outer array. A list of arrays therefore raised
`TypeError: Object of type ndarray is not JSON serializable`. The failure occurs
through `public_hash`, `begin`, `sdk_call` and the `make_carnot_agent` wrapper.
The bounded diagnostics retain the exception message, site and value types.
Historical array dtype was not retained; reconstruction does not claim that dtype.

The correction recursively converts arrays, NumPy scalar values, lists and enums.
It preserves the original public hash fields and hashes ordinary JSON as before.
It creates a fresh action context before serialization while holding the request
lock. A failed context has a null pre-observation hash and an explicit error.
Its requests remain unknown and cannot borrow the preceding action's cause.

Selection by `choose_action` and selection confirmation by `sdk_call` have
different callback owners. Repeated returns retain their invocation identity.
Equal pixels at successive actions remain distinct observations. Duplicate
requester invocations and conflicting requesters remain visible. Recording stays
opt-in, storage stays bounded and SDK exceptions keep their original propagation.

The engineering witness is exactly two fresh official SDK episodes: recording
off/on for ar25-0c556536, seed11, at most16 actions and60 seconds each. It uses
the existing live cascade and public SDK calls. Adapters, stored routes, cached
action-effect memory and language-model induction are withheld. The numerical
frame-change CNN and its checkpoint are authenticated separately. `MODEL_SPECS`
is empty; current language-model calls are zero. Imported generator identities
describe historical producers only.

Readiness requires exact action, Python/NumPy/Torch RNG, SDK, observation,
termination and legacy-observer parity, zero recorder errors and matching reset
request/action/post-observation joins. It also requires current coverage, original
semantic controls, private E2E-018/021, strict typing, lint and fresh replay.
Raw coverage SQLite, JUnit, commands, logs and source snapshots remain in the
invocation directory before copies are combined. Rehashed substitutions and
missing operands must fail replay. Publication validates final bytes through the
unchanged readers, adversarial verifier and strict row lint.

The terminal artifact is
`results/experiment_8663_v743_arc_reset_provenance.json`. Its measured fields and
bound validation sidecar are the result authority. This bounded witness gives no
solve, policy benefit, generalization or promotion credit. Exp8664 owns the full
fixed transfer panel. Same-roadmap artifacts remain future dependencies. Shared
status, changelog and traceability reconciliation belongs to the conductor.

```sh
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8663_v743_arc_reset_provenance.py --date 20261019
```
