# Cross-call induction memory

Status: implemented; local-model A/B deferred pending operator authorization.

Implement REQ-ARC-WMTE-7040 through REQ-ARC-WMTE-7042 on the scored induction
path. Rank refutation reuse first, bounded source reuse second, measurement
plumbing third. Use deterministic compaction. Keep the feature off by default.
Do not run a GPU experiment. Prove each new behavior with call-site mutations.

Defer opaque provider state, tool-loop integration, disk restart recovery, and
model-written hypothesis summaries. These need separate evidence or interfaces.

Decentralization: the mechanism uses the existing local open-weight generator.
It adds no external service or model dependency.

Validation: 40 CPU request tests; 58 distinct call-site mutations produced assertion RED,
byte-identical `cmp` restores and GREEN. See the dated research note for final
regression/collection results, limitations, and the proposed A/B population.

2026-09-05 commit-input repair: REQ-ARC-WMTE-6642 exposes the existing
`--runs-dir` through `CARNOT_ARC_EVAL_RUNS_DIR` so worktree hooks can read the
existing main-checkout corpus. Source checks stay local; missing evidence and
absent fields still refuse. Seven new tests and six assertion mutations prove
the input selection without evidence writes or a hook bypass.
