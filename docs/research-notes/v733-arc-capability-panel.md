# V733 ARC capability panel

This experiment measures whether runtime capability gating avoids attempts
that the current environment explicitly disables. It does not measure better
reasoning, hidden-game performance, or a newly discovered solve.

Exp8536 supplies the sealed qualification and frozen protocol. The panel uses
ar25-0c556536, ft09-0d8bbf25, lf52-271a04aa and tn36-ef4dde99; seeds11/22; and
baseline_applied/capability_masked_applied. Each of sixteen fresh episodes keeps
the original512-action/120-second budget. Both arms run make_carnot_agent with
LLM induction disabled. Stored solutions, routes, adapters, cached game-specific
memory and source introspection are withheld. The numeric action scorer is the
same shipped CPU component in both arms. MODEL_SPECS is empty and current LLM
calls are zero. Imported generator history remains historical provenance.

The primary engineering endpoint is selected reinduction with an observed
attempt marked disabled_by_env. Selection counts, associated skipped attempts
and all actual skipped attempts remain separate. The first disabled induction
attempt is not automatically a supervisor selection. Primitive attempt streams,
visible observations, actions, before/after effects and capability predicates
allow independent reconstruction. Exhaustion windows distinguish unavailable
arms from arms actually tried.

Each seed forms a matched baseline/masked pair. Levels, actions to progress,
complete child wall cost, unique visible pixel hashes and resolved redirects
are separate outcomes. The two seeds average within each game; the four games
are the transfer units. Game means and ranges are descriptive uncertainty.
There is no pooled action-level significance test. Missing progress times,
zero observed progress and censored redirect windows remain explicit.

A useful-efficiency finding requires fewer actual disabled attempts,
nondecreasing levels in every paired episode and no complete wall-cost increase.
The readiness score certifies collection and validation independently of this
benefit score. Mechanical eligibility success alone establishes gating only.
An honest qualified null is terminal and does not authorize outcome-driven
seed reruns, larger action budgets or game-specific helpers.

Arm refinement still requires three overlapping games and five resolved
observations per shared arm/mode cell. Unresolved outcomes cannot train arm
priorities. Official score remains unavailable. Registry exposure is recorded
before outcomes; incidental progress requires live-agent provenance and a
separate reproduction before credit. Duplicate solves cannot be headlines.

The next model-enabled study must first qualify a separate mandated Qwen
runtime with declared MODEL_SPECS and live generator receipts, then freeze a
fresh protocol. The current model-disabled panel supplies no generator benefit.

Validation uses traced unit tests,100 percent newly owned statements, private
E2E-018/021, scoped Ruff and strict mypy, and the unchanged publication,
adversarial and row-consistency consumers. Original test IDs are collected
before bounded disjoint serial shards. Fresh processes replay primitives and
reject missing evidence, owned errors and rehashed tampering. Exact final bytes
are published atomically with a bound terminal sidecar and actual downstream
consumer receipts. Prior artifacts and their sidecars remain immutable.

## Measured panel,20261015

All sixteen episodes completed512 actions without errors. Collection took
140.806 seconds. Baseline selected eight reinductions and observed eight
associated disabled attempts. Masking selected none. Counting the initial
induction too, actual disabled attempts fell from16 to8. There were8176
per-action capability exclusions. These repeated eligibility rows are not
independent observations.

Every episode remained at zero levels. All24 selected redirect observations
were right-censored and none resolved. No actions-to-progress value exists.
The complete child wall costs were69.186903 seconds for baseline and69.861957
seconds for masking. The no-increase wall-cost gate fails. The useful-efficiency
result is therefore null despite successful mechanical exclusion.

The following changes average the two masked-minus-baseline pairs within each
game. Wall differences describe this invocation; they are not timing claims
about future hardware or hidden games.

| Game | Disabled attempts | Selected disabled attempts | Levels | Actions | Visible states | Complete wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar25 | -1 | -1 | 0 | 0 | +1 | +0.053435 |
| ft09 | -1 | -1 | 0 | 0 | 0 | +0.184983 |
| lf52 | -1 | -1 | 0 | 0 | 0 | -0.023983 |
| tn36 | -1 | -1 | 0 | 0 | 0 | +0.123092 |

Readiness depends on terminal validation, separately from these observations.
The final artifact and its bound sidecar carry the authoritative classification:
`results/experiment_8537_v733_arc_capability_panel.json` and
`results/raw/experiment_8537_v733_arc_capability_panel/terminal_validation.json`.
Primitive evidence belongs to invocation1792043962122333180. The original
Exp8536 producer hash is
a02db3b46baaed4ff39320bee359a8e905dfd9c809eb299592a4dcc90ce0294f.
Ops/status/changelog/traceability reconciliation remains conductor-owned.

The initial terminal cold replay exceeded its90-second child deadline and was
killed at90.040688 seconds. This is failed owned validation. The final attempt
is disqualified, with readiness0 and efficiency0; the measured gating counts
above do not override that failure. All49 original nodes and19 original owned
tests passed, with313/313 owned statements covered. The failure receipt and
original candidate remain immutable under the invocation directory.

A regression test now requires a180–900-second bound for full primitive replay;
the runner uses240 seconds within the original900-second validation reserve.
Only validation publication resumes. No game, seed, action budget or episode
is rerun. The original runner bytes are sealed under measurement_sources/ and
the corrected runner is separately hashed. Successful replay of the final
honest failure report does not turn the failed experiment into a qualified one.

The replay-budget regression passed. Its first coverage merge omitted the
already-combined baseline file and failed the coverage gate. That failure is
also retained. Combining the two explicit coverage files restores313/313
covered statements; no scientific episode or historical test was rerun.
