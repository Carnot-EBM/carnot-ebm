# Desert Ant Labs: on-device small models (read 2026-09-20)

**Source:** https://desertant.com/ . Pages read: home, about, inspiration, and the
Redact and Schemer model pages. Read only. No model, SDK, or weight was run or
downloaded. The model cards, docs, SDK source, and license text were not read.
Every number below is vendor-reported.

## What it is

An on-device AI lab. It sells small models that each do one job. It ships them
through native Swift, Kotlin, and JavaScript SDKs and hosts weights on Hugging
Face. The catalog includes speech recognition, speech enhancement, PII
redaction, structured extraction, content moderation, and language
identification. The about page frames the models as a "cerebellum": fast,
non-reasoning, reliable, with the cloud left for reasoning.

## Claims, with what limits each one

| Claim | Where | Limit |
|---|---|---|
| Redact catches 88.8% of personal data, 99.6% precision, 23M parameters, 12MB on Apple | Redact page | Same-harness comparison, but the harness and data are theirs. The one higher-recall system named is 2.3GB. |
| Schemer scores 0.800 on unseen schemas, 0.911 absence detection vs 0.18-0.43 for prompted LLMs | Schemer page | "Internal evaluation on 9,021 held-out records." Closed beta. Model card and weights are said to be public. |
| Clips uses 470x less energy than Claude Sonnet (0.3kWh vs 140kWh per 100,000 videos) | About page | The page states the assumptions. Sonnet energy is estimated from published MLPerf runs plus measured tokens. Clips energy is 2.4s at an assumed 3W. Mixed methods. Parity was judged in a blind read on one task. |

Treat all of these as unaudited. None has a seed, a public artifact, or an
independent reproduction. They fail our artifact rules the same way any
marketing page does.

## What maps to Carnot

1. **Model proposes, deterministic check verifies.** Redact validates card
   numbers, IBANs, and national IDs against real checksums (Luhn, ISO-13616,
   per-country rules). A random 16-digit string is never masked as a card. This
   is the verifier-ensemble pattern in a small, shippable form.
2. **Grounded output is auditable by substring.** Schemer returns extracted
   strings as verbatim substrings with character offsets. A reader can audit an
   answer with a substring check. This is a cheap grounding verifier.
3. **Abstention as a headline metric.** Schemer reports "leaves a missing field
   empty" as its own score. Our calibrated-decision floor asks for accept,
   reject, or escalate with calibration. An abstention or absence-detection
   score is a natural first-class metric for it.
4. **Same framing as our hybrid.** A fast non-reasoning layer plus a reasoning
   layer matches the generator-plus-verifier split. Public wording is
   operator-curated, so this is context only.
5. **Possible `redact` implementation.** Decentralization rule 6 asks for a
   `data_handling_class` on closed-weight calls, with `redact` as one option.
   Redact masks and restores values locally, so it could implement that class.
   The license must be checked first.
6. **`llms.txt`.** The site publishes a machine-readable catalog of models and
   SDKs for coding agents. It is a cheap discoverability idea for our MCP and
   CLI surfaces.

## Open risks and unchecked items

- **License unknown.** The home page says every model is free up to 100k monthly
  active devices per platform. That is a usage cap, not an open license. Under
  the Decentralization rules it would be a decentralization-degraded tier. The
  `/license/` URL returned 404. The real path was not found.
- Whether the weights are open, and under what terms, is unchecked.
- Nothing here was reproduced. No claim above is evidence for Carnot.

## Next checks, if anyone picks this up

1. Find and read the license and each model card. Record the weight license.
2. Run Redact on a local sample and measure recall and precision on our own
   data before any adoption.
3. Decide whether an abstention score belongs in the calibrated-decision
   benchmark. This needs no Desert Ant code.
