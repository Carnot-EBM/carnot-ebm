# Compare local SemIf readout with Jev Tetris

Status: complete.

Requirement: REQ-JEV-TETRIS-001.

The public Jev Tetris project asks a cloud model to rank prose descriptions of
legal placements. This story keeps its engine, descriptions, baselines, and
seeded harness unchanged. It replaces the cloud request with a loopback server.

The server loads one cached open-weight Qwen GGUF. It reads option-token logits
after one prompt evaluation. It runs five evaluations per piece in composite
mode. The evaluations are sequential. Ambient danger and doomed outputs are
not game inputs, so this experiment omits them.

The option budget is 16 single-token labels. The first 15 placements use the
upstream stable order. The last label delegates to the first remaining
placement in that order. This preserves an incumbent fallback but does not let
the model distinguish later tail placements. The result must state this limit.

The experiment first reproduces the keyword and El-Tetris baselines. It then
runs fixed seeds with real and shuffled option probabilities. It records cited
cloud results separately from locally measured results. It makes no ARC claim.

The external code is evaluation-tier code. It was inspected before execution.
The six vendored files contain no subprocess, file-write, package, or network
path. The new cloud replacement accepts loopback endpoints only.

Experiment 10004 completed all twelve local-model runs. Both local readouts
performed near the shuffled controls and far below the local keyword and
El-Tetris baselines. This is a complete negative result for the declared
prompt, option budget, and overflow policy. It is not a broad model ranking.

## Overflow-policy correction

The paragraph above is the preserved first-pass result. Its stable-order
truncation was not a quality policy. Experiment 10005 measures more than 16
placements on 514 of 600 heuristic turns. The corrected server ranks by new
holes, bumpiness change, maximum height, aggregate height, cleared lines, and
then engine order before it applies the cap.

On Qwen3.8-27B seed 1, the corrected run improved from 28 pieces and 0 lines to
67 pieces and 17 lines. It still topped out before 300 pieces. This one-seed
check shows that the old overflow policy materially confounded the first pass.
It does not establish general readout quality or baseline parity.

## Corrected terminal comparison

Experiment 10006 reruns the full twelve-game matrix with the corrected policy.
It supersedes Experiment 10004 as the result to cite. Experiment 10004 remains
in place as the historical pre-fix record and now carries a corrigendum.

Qwen3.8-27B survived 67, 300, and 300 pieces on real seeds 1, 2, and 3. It
cleared 17, 112, and 116 lines. Its shuffled controls survived 20, 24, and 21
pieces and cleared no lines. The real arm reached the piece cap on two seeds.
Its 17-116 line range overlaps the cited cloud Jev range of 23-115. The two
completed seeds are also near the local keyword and El-Tetris results.

Qwen3.5-9B survived 28, 51, and 38 pieces. It cleared 1, 6, and 2 lines. Its
shuffled controls survived 22, 23, and 17 pieces and cleared no lines. This arm
remains below the cited cloud Jev range and both heuristic ranges.

The 27B result does not show a uniform capability gap. Seed 1 still failed
early, while seeds 2 and 3 reached the cap. The 9B arm shows a clear gap in this
interface. One design limit remains: quality ranking is a strong heuristic
candidate prefilter. The shuffle control shows that model ordering still
matters after that prefilter, but the experiment does not isolate their shares.
No further implementation bug was found.
