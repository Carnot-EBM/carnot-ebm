# Abstention HTTP/REST Doc Proposal

Proposal for the operator-curated integration docs:

- Add `POST /v1/score-candidates` as the minimal HTTP/REST verifier-scoring surface.
- Accept a JSON object with either `candidate` or `candidates`, optional `domain`, optional `abstention_mode`, and optional `abstention_threshold`.
- Keep abstention default-off; when omitted or false, response rows preserve the existing calibrated `score_candidates` shape without `verdict`.
- When `abstention_mode` is true, return per-candidate `verdict` (`confident` or `abstain`), `score`, certified `coverage`, certified `risk`, `delta`, `threshold`, and `threshold_source`.
- State that the default threshold is loaded from `results/experiment_3771_certified_abstention_operating_point.json`; threshold overrides are explicit operator choices.
- Exp 3810 repairs the blocked Exp 3801 smoke by using a true below-threshold cached verifier-scoring row for the `abstain` branch.
