# Plan: exact-label error source for the calibration pilot (2026-11-07)

Status: plan only. No run was made. Nothing here is a Carnot result.

Why: the step 1 audit (`label-provenance-audit-step1-2026-11-07.md`) found zero independently labeled errors. Step 4 of the consensus plan needs at least 200. The ImplicitHop note (`implicithop-knowledge-integration-2026-11-06.md`, lesson 2) names a source.

## Design

1. Generate random graphs. Use mean degree 3, about 50 to 60 nodes, random two-letter labels. This follows the ImplicitHop setup.
2. Ask the question "name a node exactly k hops from node X" for k = 2 and k = 3. The paper reports a 60 to 91% failure rate for other models at k = 2. Qwen3.8-27B is untested, so the rate is unknown.
3. Label each answer with breadth-first search. The label comes from code. It does not come from any Carnot verifier or from the generator.
4. Run the mandated generator (`unsloth/Qwen3.8-27B-GGUF`) with no weight update. Reasoning on and off are two separate conditions. Do not pool them.
5. Record per answer: graph seed, k, answer, label, generator settings, wall time.

## Steps

1. Pilot: 50 questions at k = 2, reasoning off. CPU builds the graphs. One GPU 0 slot through the conductor. Output: the error rate. Gate: at least 20% errors, else raise k or graph size.
2. Full run: size from the pilot error rate. Need at least 200 errors, so at least 200 / error rate questions, plus a 30% margin.
3. Freeze a split before any detector sees the data. Split by graph seed, not by question.
4. Positive control: error rate must rise with k. If it does not, stop and check the labels.

## Rules and limits

- Circularity: the breadth-first search is the oracle. A detector that calls it is `execution_grounded`, `verifier_is_oracle` is true, and it is not a moat. The pilot detector must never see the graph search. The labels may train a selector. This is the Oracle-Distinctness case in CLAUDE.md.
- Distribution gap: errors on graph questions may not look like errors on FoVer traces. Not checked. The pilot calibrates a decision on this task only.
- Labels are final-answer labels. They give no step-level localization. Do not reuse this set for localization claims.
- Preconditions: the Qwen3.8-27B GGUF is cached, GPU 0 is free, and `llama_cpp.llama_supports_gpu_offload()` prints True. GPU 1 is detached. Write `blocked_*` on any failure.
- Progress lines: print one flushed line per 10 questions so the conductor does not kill the task at 1200 seconds.
- Size: each question is one short generation. A 50-question pilot should take minutes. The full run size depends on the pilot.
- No external publication. ARC stays suspended.

## Open decision for the operator

Run the pilot through the conductor (needs a roadmap task) or by hand on GPU 0 from the outer loop. GPU 0 belongs to the conductor under the 2026-06-27 allocation, and the outer loop owns GPU 1, which is detached. The conductor route fits the allocation. I have not queued anything.

## Addendum 2026-11-07: decision recorded (append-only)

The operator answered "Yes" to the conductor route. The pilot is queued as a priority entry in `ops/known-issues.md` (in HEAD via conductor commit `ba944dec5e`). The planner has not drafted the task yet. Nothing has run. The open decision above is closed.
