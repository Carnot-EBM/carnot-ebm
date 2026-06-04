# Abstention CLI Batch Doc Proposal

Proposal for the operator-curated CLI docs:

- Add `carnot verify-batch --candidates-file <path> --domain math` as the batch verifier-scoring CLI surface.
- State that `<path>` may be a JSON array, JSONL candidate objects, or one raw candidate text per non-empty line.
- Document that abstention remains default-off; add `--abstention-mode` to emit `abstention_verdict`, `route_to_review`, and `certified_abstention` metadata.
- Note that the default threshold is loaded from `results/experiment_3771_certified_abstention_operating_point.json`; `--abstention-threshold` is an explicit operator override.
- Include a two-row example showing one `confident_error` row and one `uncertain / route to review` row.
