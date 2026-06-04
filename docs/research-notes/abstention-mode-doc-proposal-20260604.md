# Abstention Mode Documentation Proposal

Proposed operator-curated docs update for `score_candidates`:

- Add optional `abstention_mode` (default `false`).
- Add optional `abstention_threshold` override for operator tuning.
- When enabled, rows with `abstention_score` at or above `0.733216` return a confident verdict.
- Rows below the threshold return `uncertain / route to review`.
- The default threshold source is Exp 3771's certified artifact.
