# Live Telemetry Adversarial Validity Audit

Run date: `20260507`

## Verdict

- Telemetry validity verdict: `invalid_for_headline_claim_superficial_or_mechanical_gate`
- Claim allowed: `false`
- Honest verdict: `telemetry_claim_blocked_adversarial_audit`

## Confound Checks

- Length/token count: **FAIL**; best baseline `response_char_length` oriented AUROC `0.722222`.
- JSON/schema or exact-answer format: **PASS**; best baseline `exact_answer_format` oriented AUROC `0.500000`.
- Prompt family: **PASS**; best baseline `prompt_family_fover` oriented AUROC `0.500000`.
- Mock/live logprob leakage: **PASS** for telemetry baselines; BEAVER label clear `true` with mode `live_exp1468`.

## BEAVER-Lite

- Bound is sound: `true`.
- Surface constraint only: `true`.
- Single logged completion gate: `true`.
- Can pass without real verifier signal: `true`.

## Claim Boundary

The audited artifacts are useful as telemetry plumbing and a deterministic bound smoke, but they do not support a headline claim that live logprob telemetry measured a robust verifier signal.
