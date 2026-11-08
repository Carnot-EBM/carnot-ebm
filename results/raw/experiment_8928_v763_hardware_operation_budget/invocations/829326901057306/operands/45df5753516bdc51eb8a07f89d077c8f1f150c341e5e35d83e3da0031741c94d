# ImplicitHop: reading note (2026-11-06)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source: "ImplicitHop: Evaluating knowledge integration across context and memory", arXiv 2610.41169 v1 (2026-10-30), Detkov, Karish, Oleynikov, Thomson (Caltech). Read from the PDF text. The paper has 6 pages and no appendix. The figures were not read. Code release and replications were not checked.

Access note: `curl -sL -A "Mozilla/5.0" -o p.pdf https://arxiv.org/pdf/2610.41169` returns the PDF. `pdftotext -layout` extracts it.

## What the paper does

- It tests 15 open models with reasoning off. The models give no chain of thought.
- Benchmark 1 is a synthetic graph. It is an Erdos-Renyi graph with mean degree 3, random two-letter labels and about 50 to 60 nodes. The graph is an edge list in the prompt. Four tasks run on the nodes exactly k hops away (k = 0 to 3): verify, count, name one, enumerate.
- Benchmark 2 uses Wikidata. It has 30 single-valued relations and chains of 1 to 6 hops, with 200 chains per k. Shortcut chains are filtered out. Facts sit in context under invented names with 16 decoy chains, or in memory only, or in both. For the memory setting, every 1-hop fact is first checked as known.

## Reported results

| Setting | Result |
|---|---|
| Graph, name a node at k=1 | 99 to 100% |
| Graph, name a node at k=2 | 9 to 40% |
| Graph, name a node at k=3 | 12 to 24% |
| Graph, verify, k=1 to k=3 | 93 to 100% falls to 49 to 56% (near chance) |
| Graph, count or enumerate, from k=2 | at most 16% (count), at most 12% (enumerate) |
| Wikidata, facts in context | 100% at k=2, at least 94% to k=3 |
| Wikidata, facts in context, Kimi | 99% (k=3), 81% (k=4), 58% (k=5), 29% (k=6) |
| Wikidata, facts in memory only | 37 to 64% at k=2, at most 12% from k=4, 2 to 6% at k=6 |
| Wikidata, context plus memory, k=6 | 36 to 71%, against 4 to 29% for context alone |

The 12 other models, from 0.8B to 2.8T parameters, follow the same pattern.

## Limits

- Reasoning is off in every run. The paper says nothing about chain of thought or tool use, so it does not predict how the mandated generator behaves with reasoning on.
- The graph task asks for a node at exactly k hops. This is a shortest-path question, harder than plain reachability. Part of the collapse at k=2 may come from this task design. This is an outer-loop inference, not a claim in the paper.
- The two benchmarks do not compare directly. In-context Wikidata chains reach 100% at k=2. Graph "name one" falls to 9 to 40% at k=2.
- 200 chains per k. The outer loop saw no seeds or error bars in the text. The number of graphs per cell is not stated.
- The model name for one DeepSeek variant differs between the Figure 1 and Figure 2 legends ("V4 Flash" and "V4.1 Flash"). Other model names were not checked.
- The paper claims implications for prompt-based control of agents and for continual learning. It tests neither.
- The paper makes no claim about verifiers, constraints or calibration.

## What it means for Carnot

1. **Motivation, weak.** Models fail at implicit composition even when every fact is in the prompt. This supports explicit step checks in principle. The paper tests no verifier.
2. **Possible source of labeled errors for the calibration pilot.** The consensus plan (`jepa-two-model-consensus-plan-2026-11-05.md`, step 4) needs at least 200 independently labeled errors. The corpus has 114. A breadth-first search gives exact labels for random-graph questions, and models fail them often. The labels come from code, not from a Carnot verifier. Cost: a generator run, so a GPU slot through the conductor. Not checked: whether the failures resemble the errors the pilot must detect.
3. **Positive control.** Accuracy falls with k. A confidence or energy signal should fall with k too. A signal that does not is suspect.
4. **Depth stratification, hypothesis only.** Check whether detector accuracy on FoVer traces decays with the number of dependent steps. CPU only. It links to the cost-monotonicity audit (step 2 of the plan). No result exists.
5. **Benchmark design is reusable.** Random labels, chance baselines, decoys, shortcut filtering, and a check that each fact is known before it is used. The last item is the same idea as the label-provenance audit (step 1).
6. **Facts in context beat facts in weights.** This is weak, indirect support for verifiers that receive explicit facts. The paper tests a different task.

## Rule checks

- Circularity / Oracle-Distinctness: a k-hop or reachability check is exact. The breadth-first search is the oracle that defines correctness. A verifier that uses it is `execution_grounded` and `verifier_is_oracle` is true. It is not a moat and may not flip a gate. A learned selector that never calls the search may use the labels.
- No weight update to the mandated generator.
- ARC stays suspended. No ARC task follows.
- Hardware: item 2 above needs a generator run. Reserve a GPU slot through the conductor. GPU 1 is detached.

## Verdict

Low to moderate value. Background and motivation, plus one idea (item 2) that could unblock step 4 of the plan. No experiment follows from this note. Step 1 of the plan (label-provenance audit) stays first.
