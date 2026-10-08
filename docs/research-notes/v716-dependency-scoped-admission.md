# V716 dependency-scoped admission preregistration

Frozen before measurement on 2026-10-08.

Manifest: `/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8291_v716_dependency_scoped_admission/invocations/1791464085823956992/manifest.json`

SHA-256: `sha256:ed877d1c4be9de22600460b602d7e6ae4e8dc70d1daad7e2e22caf4398eea944`

Seeds 7161/7162/7163; 32/128 hard constraints; chain, sparse DAG, cyclic and dense; 24 graph units, 64 proposed updates and eight retention clock issues each. Boolean value checks, equality propagation and any-antecedent implication have explicit operands and read/write metadata. Hard constraints are immutable. Feedback delay=8. Retention nodes never receive updates. Five paired repetitions alternate arm order after one uncounted warm-up. Children are killed at issues 24 and 48 before release.

H3: zero event decision/state/admissibility disagreements with independent full rescan; exact crash parity; planted transitive conflict detected. One-hop must miss a conflict. Sparse median constraint fraction <=0.5 and paired median transaction ratio <=1 are required for efficiency. Dense/cyclic/fallback costs remain visible. No threshold tuning. Whole transactions include issue/release fsync; separate cost-receipt instrumentation is outside the committed transaction. CPU cap=600s; validation cap=900s. Frozen validation argv is adjacent validation_commands.json. Generator frozen; no_model_load; MODEL_SPECS=[]; zero LLM calls. Software fixtures alone; both generalization scores zero. Natural adoption requires another preregistration. [GRACE](https://arxiv.org/html/2607.09175v2) motivates typed locality but supplies no Carnot soundness theorem.

The first production timing attempt is preserved under raw invocation
`1791463453516975493/invalidated_attempt/` and excluded: the full-rescan wrapper
also traversed dependency closure. Its parent exited 143 after owned termination;
its source snapshots and child identities remain in the invalidation receipt.
The current rerun adds a traced zero-closure-pass baseline regression and freezes
code hashes before measurement. Graphs, event streams and thresholds are unchanged.
