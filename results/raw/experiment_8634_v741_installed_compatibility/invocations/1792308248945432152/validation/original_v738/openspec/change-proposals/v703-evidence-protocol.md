# V703 evidence protocol, frozen 2026-10-04

Exp8124 is a no-model administrative qualification. No generator is loaded.
Original Exp8098 roles and masks remain fixed. Evaluator targets are authenticated
only after the public method freeze. Historical stream observations remain usable
independently of an optional capture runtime block.

The canonical protocol below freezes exact prompt prefixes. Append a JSON object
with complete UTF-8 source and answer strings. Order alternates by the low bit of
the source-cluster SHA-256. One call per arm per source; never retry or replace
because of labels, parsing or context length. Over6000 input tokens excludes the
complete source, without truncation. Quote offsets are half-open UTF-8 byte offsets.
Probability parsing is strict JSON, finite numeric [0,1], no booleans. Quote absent
or invalid retains the probability and records validity0/ratio0. Invalid probability
makes the pair missing. Exact quote equality proves custody only; entailment is null.

```json
{"version":"v703-evidence-8124","arms":["holistic","source_span"],"calls_per_source":2,"maximum_output_tokens_per_call":128,"maximum_input_tokens":6000,"maximum_quote_words":32,"prompt_prefixes":{"holistic":"Judge whether any answer claim is unsupported by the complete source. Return only JSON with p_hallucination, a number from 0 to 1. Do not use external knowledge. Input JSON follows:\n","source_span":"Judge whether any answer claim is unsupported by the complete source. Return only JSON with p_hallucination (0 to 1), quote (one verbatim source string of at most 32 words, or null), byte_start and byte_end (half-open UTF-8 byte offsets, or null). A quote is evidence to inspect, not an entailment certificate. Do not use external knowledge. Input JSON follows:\n"},"parser":"strict-json-finite-probability-utf8-half-open-v1","order":"source_cluster_sha256_low_bit","features":["holistic_logit","overlap_mean","overlap_maximum","negation_mismatch_mean","negation_mismatch_maximum","numeric_mismatch_mean","numeric_mismatch_maximum","uncovered_fraction_mean","uncovered_fraction_maximum","span_logit","valid_quote","quote_source_byte_ratio"],"probability_clip":[1e-06,0.999999],"normalization":"fit-only mean/std; zero std becomes 1","quote_absence":[0,0],"semantic_entailment_label":null,"exposure_scope":"exposed_development_within_run_disjoint","delayed_memory_authority":"openspec/change-proposals/v702-methods-and-stream-protocol.md","KAN_CL_importance_anchoring":"deferred_existing_retirement"}
```

RT4CHART2603.27752v2 Sections3.3–3.6 decompose claims, verify overlapping local
chunks, revisit full context, and aggregate claim judgments. Our two whole-answer
arms adapt only explicit source evidence. We do not reproduce decomposition,
local/global joins, enhanced labels or published performance. Original targets
stay fixed; valid spans do not prove semantic support.

Delayed capacity2606.11711 Section2 and scheduler methods charge finite pending
feedback and lost untracked rounds. Preserve V702 delay20/capacity32 and growth
64/128/192. This empirical radial learner inherits no convex regret bound or
importance-weighted scheduling guarantee. KAN-CL anchoring stays retired.
